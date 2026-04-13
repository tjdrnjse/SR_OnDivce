"""
Knowledge Distillation Super-Resolution Model (KDSRModel).

Implements FitNet-style KD pipeline:
  - Teacher (HAT or MambaIRv2): frozen, eval-mode, on-the-fly Pseudo-GT
  - Student (RepSR / HAT-S): trained with Feature-level KD + Output-level KD

Supports **Joint-Batch training** with two data streams:
  - Stream A (RealESRGANDataset): HR images + kernels → LR synthesized on GPU
    via a two-stage degradation pipeline (blur → resize → noise → JPEG).
  - Stream B (SingleLRDataset): real LR images, no degradation applied.

References:
  - FitNets: https://arxiv.org/abs/1412.6550
  - RepDistiller: https://github.com/HobbitLong/RepDistiller
  - Real-ESRGAN: https://arxiv.org/abs/2107.10833

YAML keys consumed by this model (under `train:`):
  kd_feat_opt:
    loss_weight: 1.0     # weight for feature-level MSE loss
  kd_output_opt:
    loss_weight: 1.0     # weight for output-level L1 loss
  pixel_opt:             # optional supervised loss for Stream A vs real GT
    loss_weight: 1.0

  teacher_feat_channels: 64   # channels at teacher's conv_before_upsample output
  student_feat_channels: 64   # channels at student's conv_body output

YAML keys for tiling (under root level):
  tile:
    patch_size: 256    # LR tile spatial size
    overlap_size: 32   # LR overlap between adjacent tiles

YAML keys for Joint-Batch degradation pipeline (under `degradation:`):
  gt_size: 256                # HR patch size for Stream A (LQ = gt_size // teacher_upscale)
  resize_prob: [0.2, 0.7, 0.1]
  resize_range: [0.15, 1.5]
  gaussian_noise_prob: 0.5
  noise_range: [1, 30]
  poisson_scale_range: [0.05, 3.0]
  gray_noise_prob: 0.4
  jpeg_range: [30, 95]
  # -- second degradation --
  second_blur_prob: 0.8
  gaussian_noise_prob2: 0.5
  noise_range2: [1, 25]
  poisson_scale_range2: [0.05, 2.5]
  gray_noise_prob2: 0.4
  jpeg_range2: [30, 95]
"""

import math
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY, ARCH_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.logger import get_root_logger

from tqdm import tqdm
from os import path as osp


def _make_blend_weights(size: int, overlap: int, device) -> torch.Tensor:
    """Build a 2-D linear-ramp blending weight mask of shape (size, size).

    Values ramp from ``1/(fade+1)`` at the outermost ``fade`` pixels to
    1 in the central safe zone, where ``fade = max(1, overlap // 2)``.
    Weights are always > 0, avoiding division-by-zero when the accumulated
    weight map is inverted.

    Args:
        size (int): Spatial size of the HR patch (patch_size * scale).
        overlap (int): HR-space overlap (overlap_size * scale).
        device: Target device.

    Returns:
        Tensor: (size, size) float32 weight mask.
    """
    fade = max(1, overlap // 2)
    w = torch.ones(size, dtype=torch.float32, device=device)
    if size > 2 * fade:
        ramp = torch.linspace(
            1.0 / (fade + 1), float(fade) / (fade + 1), fade, device=device
        )
        w[:fade] = ramp
        w[size - fade:] = ramp.flip(0)
    return w.unsqueeze(0) * w.unsqueeze(1)   # outer product -> (size, size)


def tiled_inference(model: nn.Module,
                    lq: torch.Tensor,
                    patch_size: int,
                    overlap: int,
                    scale: int) -> torch.Tensor:
    """Tile-based SR inference with linear-blend stitching.

    Handles arbitrary input sizes by:
      1. Reflection-padding the LR input so its dimensions fit the tile grid.
      2. Splitting into overlapping LR tiles (stride = patch_size - overlap).
      3. Running ``model`` on each tile to obtain HR tiles.
      4. Accumulating HR tiles weighted by a 2-D linear-ramp mask (higher
         weight in the centre, lower at the overlap borders).
      5. Dividing by the accumulated weight map to normalise.
      6. Cropping the output to the exact target size (H*scale, W*scale).

    The LR overlap ``overlap`` maps to ``overlap * scale`` in HR space.
    The blending mask tapers the border region of each HR tile so that
    overlapping contributions merge seamlessly without stitching artefacts.

    Args:
        model (nn.Module): SR model (no_grad applied internally).
        lq (Tensor): (B, C, H, W) LR input tensor.
        patch_size (int): LR tile spatial size.
        overlap (int): LR-space overlap between adjacent tiles.
        scale (int): SR upscale factor.

    Returns:
        Tensor: (B, C, H*scale, W*scale) stitched HR output.
    """
    B, C, H, W = lq.shape
    stride = patch_size - overlap

    # ---- Compute reflection padding to align image to tile grid ----------
    def _pad_len(size: int) -> int:
        if size <= patch_size:
            return patch_size - size           # ensure at least one full tile
        remainder = (size - patch_size) % stride
        return (stride - remainder) % stride   # round up to next tile boundary

    pad_h = _pad_len(H)
    pad_w = _pad_len(W)
    # reflect mode requires pad < dimension; fall back to replicate when padding
    # equals or exceeds the image size (e.g. input smaller than patch_size).
    pad_mode = 'reflect' if pad_h < H and pad_w < W else 'replicate'
    lq_pad = F.pad(lq, (0, pad_w, 0, pad_h), mode=pad_mode)
    _, _, H_pad, W_pad = lq_pad.shape

    # ---- Build tile starting positions (LR space) ------------------------
    ys = list(range(0, H_pad - patch_size + 1, stride))
    xs = list(range(0, W_pad - patch_size + 1, stride))
    if ys[-1] + patch_size < H_pad:     # safety: cover the last row
        ys.append(H_pad - patch_size)
    if xs[-1] + patch_size < W_pad:     # safety: cover the last column
        xs.append(W_pad - patch_size)

    # ---- Accumulators in HR space ----------------------------------------
    hr_patch = patch_size * scale
    out_H = H_pad * scale
    out_W = W_pad * scale

    output: torch.Tensor | None = None
    weight: torch.Tensor | None = None

    blend = _make_blend_weights(hr_patch, overlap * scale, lq.device)
    blend = blend.unsqueeze(0).unsqueeze(0)   # (1, 1, hr_patch, hr_patch)

    # ---- Forward each tile and accumulate --------------------------------
    model.eval()
    for y in ys:
        for x in xs:
            tile = lq_pad[:, :, y:y + patch_size, x:x + patch_size]

            with torch.no_grad():
                sr_tile = model(tile)          # (B, C_out, hr_patch, hr_patch)

            if output is None:
                C_out = sr_tile.shape[1]
                output = lq.new_zeros(B, C_out, out_H, out_W)
                weight = lq.new_zeros(1, 1, out_H, out_W)

            y_hr = y * scale
            x_hr = x * scale
            output[:, :, y_hr:y_hr + hr_patch, x_hr:x_hr + hr_patch] += (
                sr_tile * blend
            )
            weight[:, :, y_hr:y_hr + hr_patch, x_hr:x_hr + hr_patch] += blend

    # ---- Normalise and crop to original target resolution ----------------
    output = output / weight.clamp(min=1e-8)
    return output[:, :, :H * scale, :W * scale]


@MODEL_REGISTRY.register()
class KDSRModel(SRModel):
    """Knowledge Distillation SR model.

    Inherits from SRModel so that standard BasicSR training infrastructure
    (optimizer, scheduler, saving/loading, logging) works out of the box.

    The teacher model is built from `opt['network_teacher']` and its
    pretrained weights are loaded from `opt['path']['pretrain_network_teacher']`.
    The student model uses `opt['network_g']` (same key as SRModel).
    """

    def __init__(self, opt):
        # super().__init__ calls init_training_settings() when is_train=True.
        # Our override of init_training_settings deliberately skips
        # setup_optimizers/setup_schedulers so they can be called below,
        # after feat_projector is built.
        super().__init__(opt)
        logger = get_root_logger()

        # ------------------------------------------------------------------ #
        # Build teacher                                                        #
        # ------------------------------------------------------------------ #
        teacher_opt = opt.get('network_teacher', None)
        if teacher_opt is None:
            raise ValueError(
                'KDSRModel requires `network_teacher` in the YAML config.'
            )
        self.net_teacher = ARCH_REGISTRY.get(teacher_opt['type'])(**{
            k: v for k, v in teacher_opt.items() if k != 'type'
        })
        self.net_teacher = self.model_to_device(self.net_teacher)

        # Load teacher weights
        teacher_ckpt = opt.get('path', {}).get('pretrain_network_teacher', None)
        if teacher_ckpt:
            self._load_teacher(teacher_ckpt)
        else:
            logger.warning(
                'No pretrained teacher checkpoint specified. '
                'Teacher weights are randomly initialized.'
            )

        # Freeze teacher completely
        for p in self.net_teacher.parameters():
            p.requires_grad_(False)
        self.net_teacher.eval()

        # ------------------------------------------------------------------ #
        # KD mode flags and loss weights                                       #
        # ------------------------------------------------------------------ #
        train_opt = opt.get('train', {})
        self.kd_feat_weight   = float(
            train_opt.get('kd_feat_opt',   {}).get('loss_weight', 0.0)
        )
        self.kd_output_weight = float(
            train_opt.get('kd_output_opt', {}).get('loss_weight', 0.0)
        )
        # A component is active when its loss_weight > 0
        self.use_kd_feat   = self.kd_feat_weight   > 0
        self.use_kd_output = self.kd_output_weight > 0
        if not self.use_kd_feat and not self.use_kd_output:
            logger.warning(
                'KDSRModel: both kd_feat_opt and kd_output_opt have '
                'loss_weight=0 (or are absent). Only pixel_opt will '
                'contribute to the training loss.'
            )

        # ------------------------------------------------------------------ #
        # Feature hooks (only when feature-level KD is active)                #
        # ------------------------------------------------------------------ #
        self._teacher_feat = None
        self._student_feat = None
        if self.use_kd_feat:
            self._register_feature_hooks()

        # ------------------------------------------------------------------ #
        # Feature projector — student channels → teacher channels              #
        # Built only when feature-level KD is active.                         #
        # ------------------------------------------------------------------ #
        if self.use_kd_feat:
            student_feat_ch = train_opt.get(
                'student_feat_channels',
                opt['network_g'].get('num_feat', 64)
            )
            teacher_feat_ch = train_opt.get('teacher_feat_channels', 64)
            self.feat_projector = nn.Conv2d(
                student_feat_ch, teacher_feat_ch, kernel_size=1, stride=1, padding=0
            )
            self.feat_projector = self.model_to_device(self.feat_projector)
        else:
            self.feat_projector = None

        # ------------------------------------------------------------------ #
        # GPU-side degradation pipeline (Stream A: RealESRGAN-style)          #
        # Only built when `degradation:` section is present in YAML.          #
        # ------------------------------------------------------------------ #
        self._has_degradation = 'degradation' in opt
        if self._has_degradation and self.is_train:
            self._build_degradation_pipeline()

        # ------------------------------------------------------------------ #
        # Optimizer + scheduler (deferred from init_training_settings so that  #
        # feat_projector is available when setup_optimizers runs)              #
        # ------------------------------------------------------------------ #
        if self.is_train:
            self.setup_optimizers()
            self.setup_schedulers()

        feat_str   = f'on(w={self.kd_feat_weight})'   if self.use_kd_feat   else 'off'
        output_str = f'on(w={self.kd_output_weight})' if self.use_kd_output else 'off'
        logger.info(
            f'KDSRModel | teacher: {teacher_opt["type"]} | '
            f'student: {opt["network_g"]["type"]} | '
            f'kd_feat={feat_str} | kd_output={output_str} | '
            f'degradation_pipeline={self._has_degradation}'
        )

    # ---------------------------------------------------------------------- #
    # Training settings init (override to fix two parent bugs)                #
    # ---------------------------------------------------------------------- #

    def init_training_settings(self):
        """Override SRModel.init_training_settings for two reasons:

        1. The parent raises ValueError when both pixel_opt and perceptual_opt
           are absent — but KDSRModel always has KD losses, so the check is wrong.
        2. The parent calls setup_optimizers() here, but our override of
           setup_optimizers() references self.feat_projector which is not yet
           created at this point in __init__.  We defer those calls to the end
           of KDSRModel.__init__ instead.
        """
        self.net_g.train()
        train_opt = self.opt['train']

        # EMA model (mirrors parent logic exactly)
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}'
            )
            net_g_opt = deepcopy(self.opt['network_g'])
            net_type = net_g_opt.pop('type')
            self.net_g_ema = ARCH_REGISTRY.get(net_type)(**net_g_opt).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema, load_path,
                    self.opt['path'].get('strict_load_g', True), 'params_ema'
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # Optional supervised pixel loss vs real HR GT
        if train_opt.get('pixel_opt'):
            from basicsr.losses import build_loss
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # Perceptual loss not used in KD pipeline
        self.cri_perceptual = None

        # DO NOT call setup_optimizers() or setup_schedulers() here.
        # They are called at the end of KDSRModel.__init__ once feat_projector
        # has been created.

    # ---------------------------------------------------------------------- #
    # GPU-side degradation pipeline (Stream A)                                #
    # ---------------------------------------------------------------------- #

    def _build_degradation_pipeline(self):
        """Initialise DiffJPEG and USM-sharpener for Stream A LQ synthesis."""
        from basicsr.utils import DiffJPEG, USMSharp
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = USMSharp().to(self.device)
        logger = get_root_logger()
        logger.info('KDSRModel: two-stage degradation pipeline ready (Stream A)')

    @torch.no_grad()
    def _degrade_gt_to_lq(self, gt, kernel1, kernel2, sinc_kernel):
        """Synthesize LQ from HR GT using the two-stage degradation pipeline.

        Mirrors Real-ESRGAN's GPU-side feed_data processing.
        All randomness parameters are read from ``opt['degradation']``.

        Args:
            gt         (B, C, H_gt, W_gt): HR ground-truth patches (float32).
            kernel1    (B, 21, 21)        : First-stage blur kernels.
            kernel2    (B, 21, 21)        : Second-stage blur kernels.
            sinc_kernel(B, 21, 21)        : Final sinc filter kernel.

        Returns:
            lq    (B, C, lq_size, lq_size): Synthesized LQ patches.
            gt_usm(B, C, gt_size, gt_size): USM-sharpened GT for supervision.
        """
        from basicsr.utils.img_process_util import filter2D
        from basicsr.data.degradations import (
            random_add_gaussian_noise_pt,
            random_add_poisson_noise_pt,
        )

        dopt = self.opt['degradation']
        gt_size = dopt.get('gt_size', 256)
        teacher_up = self.opt['network_teacher']['upscale']
        lq_size = gt_size // teacher_up

        # --- Crop GT batch to gt_size (same spatial crop for all items) ----
        _, _, h, w = gt.shape
        if h > gt_size:
            top = random.randint(0, h - gt_size)
            gt = gt[:, :, top:top + gt_size, :]
        if w > gt_size:
            left = random.randint(0, w - gt_size)
            gt = gt[:, :, :, left:left + gt_size]

        # --- USM sharpening -------------------------------------------------
        gt_usm = self.usm_sharpener(gt)

        # ===== First degradation pass =======================================
        out = filter2D(gt_usm, kernel1)

        # Random resize
        updown = random.choices(
            ['up', 'down', 'keep'],
            weights=dopt.get('resize_prob', [1 / 3, 1 / 3, 1 / 3])
        )[0]
        rr = dopt.get('resize_range', [0.15, 1.5])
        scale1 = (
            random.uniform(1.0, rr[1]) if updown == 'up' else
            random.uniform(rr[0], 1.0) if updown == 'down' else 1.0
        )
        mode1 = random.choice(['area', 'bilinear', 'bicubic'])
        interp_kw = {} if mode1 == 'area' else {'align_corners': False}
        out = F.interpolate(out, scale_factor=scale1, mode=mode1, **interp_kw)

        # Noise 1
        if random.random() < dopt.get('gaussian_noise_prob', 0.5):
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=dopt.get('noise_range', [1, 30]),
                clip=True, rounds=False,
                gray_prob=dopt.get('gray_noise_prob', 0.4))
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=dopt.get('poisson_scale_range', [0.05, 3.0]),
                gray_prob=dopt.get('gray_noise_prob', 0.4),
                clip=True, rounds=False)

        # JPEG 1
        jpeg_p1 = out.new_zeros(out.size(0)).uniform_(*dopt.get('jpeg_range', [30, 95]))
        out = self.jpeger(out.clamp(0, 1), quality=jpeg_p1)

        # ===== Second degradation pass ======================================
        # Resize to target LQ resolution
        mode2 = random.choice(['area', 'bilinear', 'bicubic'])
        interp_kw2 = {} if mode2 == 'area' else {'align_corners': False}
        out = F.interpolate(out, size=(lq_size, lq_size), mode=mode2, **interp_kw2)

        out = filter2D(out, kernel2)

        # Noise 2
        if random.random() < dopt.get('gaussian_noise_prob2', 0.5):
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=dopt.get('noise_range2', [1, 25]),
                clip=True, rounds=False,
                gray_prob=dopt.get('gray_noise_prob2', 0.4))
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=dopt.get('poisson_scale_range2', [0.05, 2.5]),
                gray_prob=dopt.get('gray_noise_prob2', 0.4),
                clip=True, rounds=False)

        # Final sinc (applied with probability second_blur_prob)
        if random.random() < dopt.get('second_blur_prob', 0.8):
            out = filter2D(out, sinc_kernel)

        # JPEG 2
        jpeg_p2 = out.new_zeros(out.size(0)).uniform_(*dopt.get('jpeg_range2', [30, 95]))
        lq = self.jpeger(out.clamp(0, 1), quality=jpeg_p2).clamp(0, 1)

        return lq, gt_usm

    # ---------------------------------------------------------------------- #
    # Data feeding                                                             #
    # ---------------------------------------------------------------------- #

    def feed_data(self, data):
        """Feed a batch to the model.

        Handles three input formats:

        1. **Standard paired batch** (``PairedImageDataset``):
           ``data`` has ``'lq'`` and ``'gt'`` as stacked tensors.

        2. **LR-only batch** (``SingleLRDataset``):
           ``data`` has ``'lq'`` but no ``'gt'``.

        3. **Joint Batch** (``joint_batch_training: true``):
           ``data`` produced by ``joint_collate_fn`` — keys may be stacked
           tensors (homogeneous) or plain lists with ``None`` slots
           (heterogeneous).  ``stream_id`` encodes which samples belong to
           Stream A (0, RealESRGAN) and Stream B (1, SingleLR).

           Stream A is further split by ``use_paired_lq`` (set by
           ``RealESRGANDataset``):
             - **A-deg** (``use_paired_lq == 0``): HR-only samples — LQ is
               synthesized on-the-fly via ``_degrade_gt_to_lq``.  Loss: KD +
               pixel vs USM-sharpened GT.
             - **A-bp** (``use_paired_lq == 1``): paired LQ is loaded from
               disk (bypass).  Loss: KD + supervised pixel vs real GT.
           Stream B (``SingleLRDataset``): pre-loaded LQ only.  Loss: KD only.

           The final ``self.lq`` is concatenated as ``[A-deg | A-bp | B]``.
           ``self.n_stream_a_deg`` and ``self.n_stream_a_bp`` record the slice
           sizes so ``optimize_parameters`` can apply the correct loss per
           segment.
        """
        # ---- Detect mode ---------------------------------------------------
        is_joint = isinstance(data.get('stream_id'), torch.Tensor)

        if not is_joint:
            # ---- Standard path (unchanged behaviour) -----------------------
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            elif hasattr(self, 'gt'):
                del self.gt
            self.n_stream_a_deg = 0
            self.n_stream_a_bp  = 0
            return

        # ---- Joint-Batch path ----------------------------------------------
        stream_ids = data['stream_id']          # LongTensor (B,)
        B_total = len(stream_ids)
        mask_a = (stream_ids == 0)              # boolean mask for Stream A
        mask_b = (stream_ids == 1)              # boolean mask for Stream B
        n_a = int(mask_a.sum().item())
        n_b = int(mask_b.sum().item())

        # Helper: extract items at boolean mask positions from a possibly-list value
        def _extract(key, mask):
            val = data.get(key)
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val[mask]
            # list with potential None entries
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            items = [val[i] for i in indices if val[i] is not None]
            return torch.stack(items) if items else None

        # ---- Build per-sub-stream global masks from use_paired_lq ----------
        # use_paired_lq == 0 → degradation path (A-deg)
        # use_paired_lq == 1 → paired bypass path (A-bp)
        # Stream B never has use_paired_lq, so only A indices are relevant.
        mask_a_deg = torch.zeros(B_total, dtype=torch.bool)
        mask_a_bp  = torch.zeros(B_total, dtype=torch.bool)
        n_a_deg = 0
        n_a_bp  = 0

        if n_a > 0:
            a_global_idx = mask_a.nonzero(as_tuple=True)[0]  # (n_a,) global indices
            upl_vals = data.get('use_paired_lq')
            if isinstance(upl_vals, torch.Tensor):
                upl_a = upl_vals[mask_a]          # (n_a,)
            else:
                # list with possible None entries for Stream B — grab A slots only
                upl_a = torch.stack(
                    [upl_vals[i] for i in a_global_idx.tolist()]
                )                                 # (n_a,)

            local_deg = (upl_a == 0)              # local bool mask within A
            local_bp  = (upl_a == 1)

            mask_a_deg[a_global_idx[local_deg]] = True
            mask_a_bp [a_global_idx[local_bp ]] = True
            n_a_deg = int(local_deg.sum().item())
            n_a_bp  = int(local_bp.sum().item())

        self.n_stream_a_deg = n_a_deg
        self.n_stream_a_bp  = n_a_bp

        # ---- Stream A-deg: on-the-fly degradation → LQ + USM-GT ------------
        lq_a_deg     = None
        gt_usm_a_deg = None
        if n_a_deg > 0:
            if not self._has_degradation:
                raise RuntimeError(
                    'Joint-Batch Stream A-deg (RealESRGANDataset degradation) '
                    'requires a `degradation:` section in the YAML config.'
                )
            gt_ad   = _extract('gt',          mask_a_deg).to(self.device)
            k1_ad   = _extract('kernel1',     mask_a_deg).to(self.device)
            k2_ad   = _extract('kernel2',     mask_a_deg).to(self.device)
            sinc_ad = _extract('sinc_kernel', mask_a_deg).to(self.device)
            lq_a_deg, gt_usm_a_deg = self._degrade_gt_to_lq(
                gt_ad, k1_ad, k2_ad, sinc_ad
            )

        # ---- Stream A-bp: paired LQ loaded from disk (supervised) ----------
        lq_a_bp = None
        gt_a_bp  = None
        if n_a_bp > 0:
            lq_a_bp = _extract('lq', mask_a_bp).to(self.device)
            gt_a_bp  = _extract('gt', mask_a_bp).to(self.device)

        # ---- Stream B: real LR images — KD only ----------------------------
        lq_b = None
        if n_b > 0:
            lq_b = _extract('lq', mask_b).to(self.device)

        # ---- Concatenate [A-deg | A-bp | B] into self.lq -------------------
        parts = [p for p in (lq_a_deg, lq_a_bp, lq_b) if p is not None]
        self.lq = torch.cat(parts, dim=0)

        # ---- Store GT references for optimize_parameters -------------------
        if gt_usm_a_deg is not None:
            self.gt_a_deg = gt_usm_a_deg
        elif hasattr(self, 'gt_a_deg'):
            del self.gt_a_deg

        if gt_a_bp is not None:
            self.gt_a_bp = gt_a_bp
        elif hasattr(self, 'gt_a_bp'):
            del self.gt_a_bp

        # Clear stale attributes from previous iterations
        for _attr in ('gt', 'gt_a'):
            if hasattr(self, _attr):
                delattr(self, _attr)

    # ---------------------------------------------------------------------- #
    # Hook registration                                                        #
    # ---------------------------------------------------------------------- #

    def _get_teacher_hook_layer(self):
        """Return the layer in the teacher from which to extract features.

        For HAT and MambaIRv2 the feature extraction point is
        `conv_before_upsample` (output: (B, 64, H, W)).
        Falls back gracefully if the attribute doesn't exist.
        """
        # Unwrap DataParallel / DistributedDataParallel if needed
        teacher = self.net_teacher
        if hasattr(teacher, 'module'):
            teacher = teacher.module

        for attr in ('conv_before_upsample',):
            layer = getattr(teacher, attr, None)
            if layer is not None:
                return layer

        raise AttributeError(
            'Cannot find `conv_before_upsample` in teacher model. '
            'Please ensure the teacher architecture exposes this attribute.'
        )

    def _get_student_hook_layer(self):
        """Return the layer in the student from which to extract features.

        Checks in priority order:
          - ``conv_body``          : RepSR student  (B, num_feat, H_lr, W_lr)
          - ``conv_before_upsample``: HAT-type student (B, 64, H_lr, W_lr)
        """
        student = self.net_g
        if hasattr(student, 'module'):
            student = student.module

        for attr in ('conv_body', 'conv_before_upsample'):
            layer = getattr(student, attr, None)
            if layer is not None:
                return layer

        raise AttributeError(
            'Cannot find `conv_body` or `conv_before_upsample` in student model. '
            'Please ensure the student architecture exposes one of these attributes.'
        )

    def _register_feature_hooks(self):
        def _teacher_hook(module, inp, out):
            self._teacher_feat = out

        def _student_hook(module, inp, out):
            self._student_feat = out

        teacher_layer = self._get_teacher_hook_layer()
        student_layer = self._get_student_hook_layer()

        teacher_layer.register_forward_hook(_teacher_hook)
        student_layer.register_forward_hook(_student_hook)

    # ---------------------------------------------------------------------- #
    # Teacher utilities                                                        #
    # ---------------------------------------------------------------------- #

    def _load_teacher(self, ckpt_path):
        logger = get_root_logger()
        logger.info(f'Loading teacher model from: {ckpt_path}')

        load_net = torch.load(ckpt_path, map_location=lambda s, _: s)
        # Support both plain state dict and BasicSR-style dicts
        if 'params_ema' in load_net:
            load_net = load_net['params_ema']
        elif 'params' in load_net:
            load_net = load_net['params']
        elif 'state_dict' in load_net:
            load_net = load_net['state_dict']

        # Strip 'module.' prefix if present
        clean = OrderedDict()
        for k, v in load_net.items():
            clean[k.replace('module.', '')] = v

        teacher = self.net_teacher
        if hasattr(teacher, 'module'):
            teacher = teacher.module
        teacher.load_state_dict(clean, strict=False)

    def _run_teacher(self, lq):
        """Run teacher model (no grad) and return its SR output.

        Supports tiling for memory-limited GPUs when `opt['tile']` is set.
        """
        self.net_teacher.eval()
        with torch.no_grad():
            if 'tile' in self.opt:
                # Pad input to multiple of window_size if teacher needs it
                teacher = self.net_teacher
                if hasattr(teacher, 'module'):
                    teacher = teacher.module
                window_size = getattr(teacher, 'window_size', 1)
                _, _, h, w = lq.shape
                pad_h = (window_size - h % window_size) % window_size
                pad_w = (window_size - w % window_size) % window_size
                lq_pad = F.pad(lq, (0, pad_w, 0, pad_h), 'reflect')

                pseudo_gt = tiled_inference(
                    self.net_teacher,
                    lq_pad,
                    self.opt['tile']['patch_size'],
                    self.opt['tile']['overlap_size'],
                    self.opt['scale']
                )
                # Remove padding
                _, _, h_out, w_out = pseudo_gt.shape
                pseudo_gt = pseudo_gt[
                    :, :,
                    :h_out - pad_h * self.opt['scale'],
                    :w_out - pad_w * self.opt['scale']
                ]
            else:
                pseudo_gt = self.net_teacher(lq)
        return pseudo_gt

    # ---------------------------------------------------------------------- #
    # Training loop overrides                                                  #
    # ---------------------------------------------------------------------- #

    def setup_optimizers(self):
        """Set up optimizer for student (and feature projector when feat KD is active)."""
        train_opt = self.opt['train']
        optim_params = [{'params': self.net_g.parameters()}]
        if self.feat_projector is not None:
            optim_params.append({'params': self.feat_projector.parameters()})
        # Copy to avoid mutating the config dict (which would break resume)
        optim_cfg = {k: v for k, v in train_opt['optim_g'].items() if k != 'type'}
        optim_type = train_opt['optim_g']['type']
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **optim_cfg
        )
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # ---- Teacher: generate Pseudo-GT and capture teacher features -------
        # We need teacher features at training resolution (small patches),
        # so we run the teacher on lq directly (no tiling needed for patches).
        # ---- Cross-scale validation -----------------------------------------
        teacher_upscale = self.opt['network_teacher']['upscale']
        student_upscale = self.opt['network_g']['upscale']
        if teacher_upscale < student_upscale:
            raise ValueError(
                f'Teacher upscale ({teacher_upscale}x) is less than student '
                f'upscale ({student_upscale}x). KD cannot be performed when '
                'the teacher resolution is lower than the student resolution.'
            )

        # ---- Teacher forward (skipped if both feat/output KD are off) -------
        pseudo_gt    = None
        teacher_feat = None
        _, _, h, w = self.lq.shape

        if self.use_kd_feat or self.use_kd_output:
            self.net_teacher.eval()
            with torch.no_grad():
                # Pad lq to window_size multiple for transformer teachers
                teacher = self.net_teacher
                if hasattr(teacher, 'module'):
                    teacher = teacher.module
                window_size = getattr(teacher, 'window_size', 1)
                pad_h = (window_size - h % window_size) % window_size
                pad_w = (window_size - w % window_size) % window_size
                lq_padded = F.pad(self.lq, (0, pad_w, 0, pad_h), 'reflect')
                pseudo_gt_padded = self.net_teacher(lq_padded)

                # Crop padding back to original LR-derived HR size
                _, _, h_out, w_out = pseudo_gt_padded.shape
                pseudo_gt = pseudo_gt_padded[
                    :, :,
                    :h_out - pad_h * teacher_upscale,
                    :w_out - pad_w * teacher_upscale,
                ]

                # Cross-scale: downsample pseudo_gt to student output resolution
                if teacher_upscale > student_upscale:
                    pseudo_gt = F.interpolate(
                        pseudo_gt,
                        size=(h * student_upscale, w * student_upscale),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True,
                    )

            # teacher_feat captured by hook during the forward above
            if self.use_kd_feat:
                # Crop padded feature back to original LR spatial size
                teacher_feat = self._teacher_feat[:, :, :h, :w].detach()

        # ---- Student: forward pass ------------------------------------------
        self.net_g.train()
        student_out = self.net_g(self.lq)

        # ---- Feature-level KD loss (FitNet MSE) — optional ------------------
        l_kd_feat = torch.zeros(1, device=self.device)
        if self.use_kd_feat:
            student_feat_proj = self.feat_projector(self._student_feat)
            # Align spatial resolution if S2D compresses spatial dims in student
            if student_feat_proj.shape[2:] != teacher_feat.shape[2:]:
                student_feat_proj = F.interpolate(
                    student_feat_proj,
                    size=teacher_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            l_kd_feat = F.mse_loss(student_feat_proj, teacher_feat) * self.kd_feat_weight

        # ---- Output-level KD loss (L1 vs teacher pseudo-GT) — optional ------
        l_kd_out = torch.zeros(1, device=self.device)
        if self.use_kd_output:
            l_kd_out = F.l1_loss(student_out, pseudo_gt) * self.kd_output_weight

        # ---- Optional supervised pixel loss --------------------------------
        # Standard batch:  self.gt covers the whole batch.
        # Joint batch: three independent slices in [A-deg | A-bp | B] order.
        #   A-deg → pixel loss vs USM-sharpened GT  (self.gt_a_deg)
        #   A-bp  → supervised pixel loss vs real GT (self.gt_a_bp)
        #   B     → no pixel loss (KD only)
        l_total   = l_kd_feat + l_kd_out
        l_pix     = torch.zeros(1, device=self.device)
        l_pix_deg = torch.zeros(1, device=self.device)
        l_pix_bp  = torch.zeros(1, device=self.device)
        if self.cri_pix is not None:
            if hasattr(self, 'gt'):
                # Standard paired batch or non-joint KD with explicit GT
                l_pix = self.cri_pix(student_out, self.gt)
                l_total = l_total + l_pix
            else:
                # Joint batch: apply per-stream pixel losses
                n_deg = getattr(self, 'n_stream_a_deg', 0)
                n_bp  = getattr(self, 'n_stream_a_bp',  0)

                # A-deg slice: pixel loss vs USM-sharpened GT
                if n_deg > 0 and hasattr(self, 'gt_a_deg'):
                    out_deg = student_out[:n_deg]
                    gt_deg  = self.gt_a_deg
                    if gt_deg.shape[2:] != out_deg.shape[2:]:
                        gt_deg = F.interpolate(
                            gt_deg, size=out_deg.shape[2:],
                            mode='bicubic', antialias=True, align_corners=False)
                    l_pix_deg = self.cri_pix(out_deg, gt_deg)

                # A-bp slice: supervised pixel loss vs paired GT
                if n_bp > 0 and hasattr(self, 'gt_a_bp'):
                    out_bp = student_out[n_deg:n_deg + n_bp]
                    gt_bp  = self.gt_a_bp
                    if gt_bp.shape[2:] != out_bp.shape[2:]:
                        gt_bp = F.interpolate(
                            gt_bp, size=out_bp.shape[2:],
                            mode='bicubic', antialias=True, align_corners=False)
                    l_pix_bp = self.cri_pix(out_bp, gt_bp)

                # Stream B: no pixel loss
                if n_deg > 0 or n_bp > 0:
                    l_pix = l_pix_deg + l_pix_bp
                    l_total = l_total + l_pix

        l_total.backward()
        self.optimizer_g.step()

        # ---- Logging -------------------------------------------------------
        self.log_dict = self.reduce_loss_dict(OrderedDict(
            l_kd_feat=l_kd_feat,
            l_kd_out=l_kd_out,
            l_pix=l_pix,
            l_pix_deg=l_pix_deg,
            l_pix_bp=l_pix_bp,
            l_total=l_total,
        ))

        # Cache first-sample visuals for TensorBoard (detached, no grad)
        self._vis_lq      = self.lq[:1].detach()
        self._vis_student = student_out[:1].detach()
        self._vis_teacher = pseudo_gt[:1].detach() if pseudo_gt is not None else None

    # ---------------------------------------------------------------------- #
    # Validation / Inference overrides (student only)                          #
    # ---------------------------------------------------------------------- #

    def pre_process(self):
        """No mandatory padding for RepSR (conv2d with 'same' equivalent)."""
        self.img = self.lq
        self.mod_pad_h = 0
        self.mod_pad_w = 0

    def process(self):
        self.net_g.eval()
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output = self.net_g_ema(self.img)
            else:
                self.output = self.net_g(self.img)

    def tile_process(self):
        """Tile-based inference for the student model.

        Uses parameters from opt['tile']['patch_size'] and
        opt['tile']['overlap_size'].
        """
        model = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        self.output = tiled_inference(
            model,
            self.img,
            self.opt['tile']['patch_size'],
            self.opt['tile']['overlap_size'],
            self.opt['scale']
        )

    def post_process(self):
        # RepSR doesn't require mandatory padding removal.
        pass

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {
                    metric: 0 for metric in self.opt['val']['metrics'].keys()
                }
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # Maximum number of (lq / teacher / student) image triplets to save
        # per validation round.  Set via val.save_img_num in YAML (default 4).
        save_img_num = int(self.opt['val'].get('save_img_num', 4))
        n_saved = 0

        metric_data = {}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])    # student output
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # -------------------------------------------------------------- #
            # Save triplet: LQ (upsampled) / Teacher output / Student output  #
            # Saved for the first `save_img_num` images per validation round. #
            # -------------------------------------------------------------- #
            if save_img and n_saved < save_img_num:
                scale = self.opt['scale']

                # LQ: bilinear upsample to HR resolution for side-by-side compare
                lq_up = F.interpolate(
                    self.lq,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False,
                )
                lq_img = tensor2img([lq_up])

                # Teacher pseudo-GT (no grad, handles window_size padding)
                teacher_out = self._run_teacher(self.lq)
                teacher_img = tensor2img([teacher_out])

                # Build save directory
                if self.opt['is_train']:
                    save_dir = osp.join(
                        self.opt['path']['visualization'],
                        dataset_name,
                        f'iter_{current_iter:08d}',
                    )
                else:
                    suffix = self.opt['val'].get('suffix', self.opt['name'])
                    save_dir = osp.join(
                        self.opt['path']['visualization'],
                        dataset_name,
                        suffix,
                    )

                imwrite(lq_img,      osp.join(save_dir, f'{img_name}_lq.png'))
                imwrite(teacher_img, osp.join(save_dir, f'{img_name}_teacher.png'))
                imwrite(sr_img,      osp.join(save_dir, f'{img_name}_student.png'))

                # Also log to TensorBoard (uses the same tensors, already computed)
                if tb_logger is not None:
                    tag_prefix = f'val/{dataset_name}/{img_name}'
                    tb_logger.add_image(
                        f'{tag_prefix}_lq_bicubic',
                        self._to_tb_image(lq_up),
                        global_step=current_iter
                    )
                    tb_logger.add_image(
                        f'{tag_prefix}_teacher',
                        self._to_tb_image(teacher_out),
                        global_step=current_iter
                    )
                    tb_logger.add_image(
                        f'{tag_prefix}_student',
                        self._to_tb_image(visuals['result']),
                        global_step=current_iter
                    )

                n_saved += 1

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(
                    dataset_name, metric,
                    self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    # ---------------------------------------------------------------------- #
    # TensorBoard visual logging                                               #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _to_tb_image(tensor: torch.Tensor, max_size: int = 256) -> torch.Tensor:
        """Convert a model output tensor to a TensorBoard-compatible image.

        Args:
            tensor: (1, C, H, W) float tensor in [0, 1] range.
            max_size: Resize longest side to this if image is larger.

        Returns:
            (C, H', W') float tensor in [0, 1], clipped.
        """
        img = tensor[0].clamp(0.0, 1.0).float()   # (C, H, W)
        _, H, W = img.shape
        if max(H, W) > max_size:
            scale = max_size / max(H, W)
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            img = F.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return img

    def log_train_visuals(self, tb_logger, current_iter: int):
        """Log training sample visuals (LQ / teacher pseudo-GT / student) to TB.

        Called from ``hat/train.py`` every ``tb_train_vis_freq`` iterations.
        Relies on ``self._vis_lq``, ``self._vis_teacher``, ``self._vis_student``
        cached at the end of the most recent ``optimize_parameters`` call.
        """
        if tb_logger is None:
            return
        if not (hasattr(self, '_vis_lq') and hasattr(self, '_vis_teacher')
                and hasattr(self, '_vis_student')):
            return

        scale = self.opt['scale']
        lq_up = F.interpolate(
            self._vis_lq, scale_factor=scale, mode='bilinear', align_corners=False
        )

        tb_logger.add_image(
            'train/lq_bicubic',
            self._to_tb_image(lq_up),
            global_step=current_iter
        )
        tb_logger.add_image(
            'train/teacher_pseudo_gt',
            self._to_tb_image(self._vis_teacher),
            global_step=current_iter
        )
        tb_logger.add_image(
            'train/student_output',
            self._to_tb_image(self._vis_student),
            global_step=current_iter
        )

    # ---------------------------------------------------------------------- #
    # Saving                                                                   #
    # ---------------------------------------------------------------------- #

    def save(self, epoch, current_iter):
        """Save student model and EMA (teacher is never saved)."""
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema], 'net_g', current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
