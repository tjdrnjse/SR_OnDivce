"""
RepSR architecture: RepVGG-based Super-Resolution network.
Adapted from: https://github.com/JL-DY/RepSR
Ported to BasicSR/HAT framework.

Features:
  - Multi-branch training (3x3 Conv, 1x1 Conv, Identity), each with BN
  - Single 3x3 Conv inference via structural reparameterization
  - Optional Space-to-Depth (nn.PixelUnshuffle) pre-processing
  - Configurable block count and channel width via YAML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class RepSRBlock(nn.Module):
    """RepVGG-style residual block with structural reparameterization.

    During training: three parallel branches (3x3+BN, 1x1+BN, Identity+BN)
    are summed before the activation.
    At inference: call reparameterize() to fuse all branches into a single
    3x3 convolution, then set self.deployed = True.

    Args:
        num_feat (int): Number of feature channels (in == out).
    """

    def __init__(self, num_feat: int):
        super().__init__()
        self.num_feat = num_feat
        self.deployed = False

        # 3x3 Conv branch
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_feat)
        )
        # 1x1 Conv branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_feat)
        )
        # Identity branch (BN only; identity kernel fused at reparameterize)
        self.branch_id = nn.BatchNorm2d(num_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.rep_conv(x)
        return self.branch_3x3(x) + self.branch_1x1(x) + self.branch_id(x)

    # ------------------------------------------------------------------
    # Reparameterization helpers
    # ------------------------------------------------------------------

    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """Fuse a Conv2d + BatchNorm2d into a single Conv2d (weight, bias)."""
        kernel = conv.weight                          # (C_out, C_in, kH, kW)
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_weight = kernel * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_weight, fused_bias

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d):
        """Fuse identity mapping + BN into a (weight, bias) pair."""
        # Create identity kernel: shape (C, C, 3, 3), center=1
        c = self.num_feat
        identity_kernel = torch.zeros(c, c, 3, 3,
                                      dtype=bn.weight.dtype,
                                      device=bn.weight.device)
        for i in range(c):
            identity_kernel[i, i, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_weight = identity_kernel * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_weight, fused_bias

    def reparameterize(self):
        """Merge all branches into a single 3×3 Conv and switch to deployed mode.

        After calling this method the module no longer holds the original
        branch parameters. This is irreversible.
        """
        if self.deployed:
            return

        # 3x3 branch
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        # 1x1 branch – pad to 3x3
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        k1 = F.pad(k1, [1, 1, 1, 1])
        # Identity branch
        kid, bid = self._fuse_identity_bn(self.branch_id)

        final_weight = k3 + k1 + kid
        final_bias = b3 + b1 + bid

        self.rep_conv = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        self.rep_conv.weight.data = final_weight
        self.rep_conv.bias.data = final_bias

        # Remove training-only parameters
        del self.branch_3x3, self.branch_1x1, self.branch_id

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
