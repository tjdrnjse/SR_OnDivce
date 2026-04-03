"""
RepSR architecture: SR-optimized reparameterizable Super-Resolution network.
Adapted from: https://github.com/JL-DY/RepSR
Ported to BasicSR/HAT framework.

Features:
  - SR-optimized multi-branch block: 2x (Conv3x3 -> BN -> Conv1x1) + Identity
  - Channel expansion (c -> 2c -> c) inside each branch for SR feature capacity
  - Single 3x3 Conv inference via structural reparameterization
  - Optional Space-to-Depth (nn.PixelUnshuffle) pre-processing
  - Configurable block count and channel width via YAML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class RepSRBlock(nn.Module):
    """SR-optimized reparameterizable block.

    During training, two symmetric branches are summed with an identity skip:
        out = x + branch1(x) + branch2(x)
    Each branch expands channels (c -> 2c -> c):
        branch(x) = Conv1x1(BN(Conv3x3(x)))
                    Conv(c->2c, 3x3) -> BN(2c) -> Conv(2c->c, 1x1)

    At inference, call reparameterize() to fuse the entire block into a
    single equivalent 3x3 Conv. The fused kernel is derived as follows:
        1. Fuse Conv3x3 + BN  ->  k_fused (2c, c, 3, 3), b_fused (2c,)
        2. Apply Conv1x1 weights to collapse 2c back to c:
               merged_w[i,k,h,w] = sum_j  W1[i,j] * k_fused[j,k,h,w]
               merged_b[i]       = sum_j  W1[i,j] * b_fused[j]  +  b1[i]
        3. Add identity kernel (1.0 at each [i,i,1,1]) for the skip branch.
        4. Sum both merged branch kernels and the identity kernel.

    Args:
        num_feat (int): Number of feature channels (in == out).
    """

    def __init__(self, num_feat: int):
        super().__init__()
        self.num_feat = num_feat
        self.deployed = False
        mid = num_feat * 2  # channel expansion factor

        # Branch 1: Conv(c->2c, 3x3, no bias) -> BN(2c) -> Conv(2c->c, 1x1, bias)
        self.branch1_3x3 = nn.Conv2d(num_feat, mid, 3, 1, 1, bias=False)
        self.branch1_bn  = nn.BatchNorm2d(mid)
        self.branch1_1x1 = nn.Conv2d(mid, num_feat, 1, 1, 0, bias=True)

        # Branch 2: identical structure, independent weights
        self.branch2_3x3 = nn.Conv2d(num_feat, mid, 3, 1, 1, bias=False)
        self.branch2_bn  = nn.BatchNorm2d(mid)
        self.branch2_1x1 = nn.Conv2d(mid, num_feat, 1, 1, 0, bias=True)

    def _branch_forward(self, x, conv3x3, bn, conv1x1):
        return conv1x1(bn(conv3x3(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.rep_conv(x)
        return (x
                + self._branch_forward(x, self.branch1_3x3, self.branch1_bn, self.branch1_1x1)
                + self._branch_forward(x, self.branch2_3x3, self.branch2_bn, self.branch2_1x1))

    # ------------------------------------------------------------------
    # Reparameterization helpers
    # ------------------------------------------------------------------

    def _fuse_branch(self, conv3x3: nn.Conv2d, bn: nn.BatchNorm2d, conv1x1: nn.Conv2d):
        """Fuse Conv3x3 + BN + Conv1x1 into an equivalent (weight, bias) pair.

        Returns:
            merged_w: (num_feat, num_feat, 3, 3)
            merged_b: (num_feat,)
        """
        # ---- Step 1: fuse Conv3x3 (c->2c) with BN(2c) ----
        # k_fused[j, k, h, w] = (gamma_j / std_j) * conv3x3.weight[j, k, h, w]
        # b_fused[j]           = beta_j - mean_j * gamma_j / std_j
        k3 = conv3x3.weight                                    # (2c, c, 3, 3)
        std = (bn.running_var + bn.eps).sqrt()                 # (2c,)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)            # (2c, 1, 1, 1)
        k_fused = k3 * t                                       # (2c, c, 3, 3)
        b_fused = bn.bias - bn.running_mean * bn.weight / std  # (2c,)

        # ---- Step 2: apply Conv1x1 (2c->c) to collapse the expanded dim ----
        # merged_w[i, k, h, w] = sum_j  W1[i, j] * k_fused[j, k, h, w]
        # merged_b[i]          = sum_j  W1[i, j] * b_fused[j]  +  b1[i]
        W1 = conv1x1.weight[:, :, 0, 0]   # (c, 2c)  – squeeze spatial dims
        b1 = conv1x1.bias                 # (c,)

        merged_w = torch.einsum('ij,jkhw->ikhw', W1, k_fused)  # (c, c, 3, 3)
        merged_b = W1 @ b_fused + b1                             # (c,)

        return merged_w, merged_b

    def reparameterize(self):
        """Merge all branches into a single 3x3 Conv and switch to deployed mode.

        After calling this method the module no longer holds the original
        branch parameters. This is irreversible.
        """
        if self.deployed:
            return

        c = self.num_feat
        dtype = self.branch1_3x3.weight.dtype
        device = self.branch1_3x3.weight.device

        # Fuse both expanded branches
        k1, b1 = self._fuse_branch(self.branch1_3x3, self.branch1_bn, self.branch1_1x1)
        k2, b2 = self._fuse_branch(self.branch2_3x3, self.branch2_bn, self.branch2_1x1)

        # Identity branch: 3x3 kernel with 1.0 at each diagonal center
        k_id = torch.zeros(c, c, 3, 3, dtype=dtype, device=device)
        for i in range(c):
            k_id[i, i, 1, 1] = 1.0
        b_id = torch.zeros(c, dtype=dtype, device=device)

        final_weight = k1 + k2 + k_id
        final_bias   = b1 + b2 + b_id

        self.rep_conv = nn.Conv2d(c, c, 3, 1, 1)
        self.rep_conv.weight.data = final_weight
        self.rep_conv.bias.data   = final_bias

        # Remove training-only parameters to save memory
        del self.branch1_3x3, self.branch1_bn, self.branch1_1x1
        del self.branch2_3x3, self.branch2_bn, self.branch2_1x1

        self.deployed = True

    def extra_repr(self) -> str:
        return (f'num_feat={self.num_feat}, '
                f'deployed={self.deployed}')


@ARCH_REGISTRY.register()
class RepSR(nn.Module):
    """RepVGG-based Super-Resolution network with optional Space-to-Depth.

    YAML configuration example::

        network_g:
          type: RepSR
          num_in_ch: 3
          num_feat: 64
          num_blocks: 8
          upscale: 4
          use_space_to_depth: false   # set true to enable S2D pre-processing
          s2d_factor: 2               # PixelUnshuffle factor (used when S2D=true)

    Architecture (training)::

        Input (B,C,H,W)
          │ [optional PixelUnshuffle(s2d_factor) when use_space_to_depth=True]
          ▼
        conv_first
          │
        [num_blocks × (RepSRBlock + PReLU)]   ← multi-branch during training
          │
        conv_body                              ← KD feature hook point
          │  + residual from conv_first
          ▼
        conv_up → PixelShuffle(effective_scale)
          │  + bilinear skip of original input
          ▼
        Output (B,C,H*upscale,W*upscale)

    Args:
        num_in_ch (int): Input channels. Default: 3.
        num_feat (int): Internal feature channels. Default: 64.
        num_blocks (int): Number of RepSR blocks. Default: 8.
        upscale (int): SR scale factor. Default: 4.
        use_space_to_depth (bool): Enable PixelUnshuffle pre-processing.
            Default: False.
        s2d_factor (int): PixelUnshuffle downscale factor.
            Only used when use_space_to_depth=True. Default: 2.
    """

    def __init__(self,
                 num_in_ch: int = 3,
                 num_feat: int = 64,
                 num_blocks: int = 8,
                 upscale: int = 4,
                 use_space_to_depth: bool = False,
                 s2d_factor: int = 2):
        super().__init__()

        self.upscale = upscale
        self.use_space_to_depth = use_space_to_depth
        self.s2d_factor = s2d_factor if use_space_to_depth else 1

        # --- Space-to-Depth pre-processing -----------------------------------
        if use_space_to_depth:
            self.pixel_unshuffle = nn.PixelUnshuffle(s2d_factor)
            first_in_ch = num_in_ch * s2d_factor * s2d_factor
            effective_scale = upscale * s2d_factor
        else:
            first_in_ch = num_in_ch
            effective_scale = upscale

        self._effective_scale = effective_scale

        # --- Stem ------------------------------------------------------------
        self.conv_first = nn.Conv2d(first_in_ch, num_feat, 3, 1, 1)

        # --- Body: RepSR blocks with PReLU -----------------------------------
        body_layers = []
        for _ in range(num_blocks):
            body_layers.append(RepSRBlock(num_feat))
            body_layers.append(nn.PReLU(num_feat))
        self.body = nn.Sequential(*body_layers)

        # --- Post-body conv (feature extraction hook for KD) ----------------
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # --- Upsampling tail -------------------------------------------------
        self.conv_up = nn.Conv2d(
            num_feat,
            num_in_ch * effective_scale * effective_scale,
            3, 1, 1
        )
        self.pixel_shuffle = nn.PixelShuffle(effective_scale)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x  # keep for skip connection

        # Space-to-Depth
        if self.use_space_to_depth:
            x = self.pixel_unshuffle(x)

        # Stem
        x = self.conv_first(x)
        res = x

        # Body
        x = self.body(x)

        # Post-body conv + residual
        x = self.conv_body(x) + res  # ← KD hook point via self.conv_body

        # Upsample
        x = self.pixel_shuffle(self.conv_up(x))

        # Bilinear skip connection from original LR input
        skip = F.interpolate(
            x_orig,
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )
        return x + skip

    def reparameterize(self):
        """Fuse all RepSRBlock multi-branches into single 3×3 Convs.

        Call this before saving the deployment-ready checkpoint.
        After reparameterization the model is functionally identical but
        all RepSRBlock layers have been replaced with a single Conv2d.
        """
        for module in self.modules():
            if isinstance(module, RepSRBlock):
                module.reparameterize()

    def extra_repr(self) -> str:
        return (f'upscale={self.upscale}, '
                f'use_space_to_depth={self.use_space_to_depth}, '
                f's2d_factor={self.s2d_factor}')
