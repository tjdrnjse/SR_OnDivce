"""
Knowledge Distillation Super-Resolution Model (KDSRModel).

Implements FitNet-style KD pipeline:
  - Teacher (HAT or MambaIRv2): frozen, eval-mode, on-the-fly Pseudo-GT
  - Student (RepSR): trained with Feature-level KD + Output-level KD

References:
  - FitNets: https://arxiv.org/abs/1412.6550
  - RepDistiller: https://github.com/HobbitLong/RepDistiller

YAML keys consumed by this model (under `train:`):
  kd_feat_opt:
    loss_weight: 1.0     # weight for feature-level MSE loss
  kd_output_opt:
    loss_weight: 1.0     # weight for output-level L1 loss
  pixel_opt:             # optional supervised loss against ground-truth HR
    loss_weight: 1.0

  teacher_feat_channels: 64   # channels at teacher's conv_before_upsample output
  student_feat_channels: 64   # channels at student's conv_body output

YAML keys for tiling (under root level):
  tile:
    patch_size: 256    # LR tile spatial size
    overlap_size: 32   # LR overlap between adjacent tiles
"""

import math
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
        # Feature hooks                                                        #
        # ------------------------------------------------------------------ #
        self._teacher_feat = None
        self._student_feat = None
        self._register_feature_hooks()

        # ------------------------------------------------------------------ #
        # Feature projector (student channels → teacher channels)             #
        # ------------------------------------------------------------------ #
        train_opt = opt.get('train', {})
        student_feat_ch = train_opt.get(
            'student_feat_channels',
            opt['network_g'].get('num_feat', 64)
        )
        teacher_feat_ch = train_opt.get('teacher_feat_channels', 64)

        self.feat_projector = nn.Conv2d(
            student_feat_ch, teacher_feat_ch, kernel_size=1, stride=1, padding=0
        )
        self.feat_projector = self.model_to_device(self.feat_projector)

        # ------------------------------------------------------------------ #
        # KD loss weights                                                      #
        # ------------------------------------------------------------------ #
        self.kd_feat_weight = train_opt.get(
            'kd_feat_opt', {}
        ).get('loss_weight', 1.0)
        self.kd_output_weight = train_opt.get(
            'kd_output_opt', {}
        ).get('loss_weight', 1.0)

        # ------------------------------------------------------------------ #
        # Optimizer + scheduler (deferred from init_training_settings so that  #
        # feat_projector is available when setup_optimizers runs)              #
        # ------------------------------------------------------------------ #
        if self.is_train:
            self.setup_optimizers()
            self.setup_schedulers()

        logger.info(
            f'KDSRModel | teacher: {teacher_opt["type"]} | '
            f'student: {opt["network_g"]["type"]} | '
            f'feat_weight={self.kd_feat_weight} | '
            f'output_weight={self.kd_output_weight}'
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
    # Data feeding                                                             #
    # ---------------------------------------------------------------------- #

    def feed_data(self, data):
        """Feed a batch to the model.

        Accepts data dicts both with and without a ``'gt'`` key so that
        ``SingleLRDataset`` (LR-only) and paired datasets can both be used.
        Stale ``self.gt`` from a previous iteration is cleared when the
        current batch has no GT, preventing cross-iteration leakage.
        """
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        elif hasattr(self, 'gt'):
            del self.gt   # clear stale GT from any previous paired batch

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
        """Set up optimizer for both student and feature projector."""
        train_opt = self.opt['train']
        optim_params = [
            {'params': self.net_g.parameters()},
            {'params': self.feat_projector.parameters()},
        ]
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

        self.net_teacher.eval()
        with torch.no_grad():
            # Pad lq to window_size multiple for transformer teachers
            teacher = self.net_teacher
            if hasattr(teacher, 'module'):
                teacher = teacher.module
            window_size = getattr(teacher, 'window_size', 1)
            _, _, h, w = self.lq.shape
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            lq_padded = F.pad(self.lq, (0, pad_w, 0, pad_h), 'reflect')
            pseudo_gt_padded = self.net_teacher(lq_padded)

            # Crop padding using teacher_upscale (not student scale)
            # pseudo_gt_padded: (B, C, (h+pad_h)*teacher_up, (w+pad_w)*teacher_up)
            # After crop:       (B, C, h*teacher_up, w*teacher_up)
            _, _, h_out, w_out = pseudo_gt_padded.shape
            pseudo_gt = pseudo_gt_padded[
                :, :,
                :h_out - pad_h * teacher_upscale,
                :w_out - pad_w * teacher_upscale,
            ]

            # Cross-scale: downsample pseudo_gt to student output resolution
            # E.g. teacher x4 → pseudo_gt (H*4, W*4) → bicubic → (H*3, W*3)
            if teacher_upscale > student_upscale:
                pseudo_gt = F.interpolate(
                    pseudo_gt,
                    size=(h * student_upscale, w * student_upscale),
                    mode='bicubic',
                    align_corners=False,
                    antialias=True,
                )
            # teacher_feat is stored in self._teacher_feat by hook

        # teacher_feat shape: (B, 64, H_lr+pad_h, W_lr+pad_w) — includes padding
        # Crop it back to original LR spatial size so we only supervise on real content.
        teacher_feat = self._teacher_feat[:, :, :h, :w].detach()

        # ---- Student: forward pass, capture student features ----------------
        self.net_g.train()
        student_out = self.net_g(self.lq)
        # student_feat shape: (B, student_ch, H_lr, W_lr) without S2D
        #                     (B, student_ch, H_lr/r, W_lr/r) with S2D
        student_feat = self._student_feat

        # ---- Feature-level KD loss (FitNet MSE) ----------------------------
        # Project student channels → teacher channels
        student_feat_proj = self.feat_projector(student_feat)

        # Align spatial resolution: after crop, teacher_feat is (B,64,H_lr,W_lr).
        # If S2D is active, student_feat_proj is at (H_lr/r, W_lr/r) — upsample
        # to match teacher resolution so MSE is computed at the same grid.
        if student_feat_proj.shape[2:] != teacher_feat.shape[2:]:
            student_feat_proj = F.interpolate(
                student_feat_proj,
                size=teacher_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        l_kd_feat = F.mse_loss(student_feat_proj, teacher_feat) * self.kd_feat_weight

        # ---- Output-level KD loss (L1 vs teacher pseudo-GT) ----------------
        l_kd_out = F.l1_loss(student_out, pseudo_gt) * self.kd_output_weight

        # ---- Optional supervised pixel loss (vs real HR GT) ----------------
        l_total = l_kd_feat + l_kd_out
        l_pix = torch.zeros_like(l_total)
        if self.cri_pix is not None and hasattr(self, 'gt'):
            l_pix = self.cri_pix(student_out, self.gt)
            l_total = l_total + l_pix

        l_total.backward()
        self.optimizer_g.step()

        # ---- Logging -------------------------------------------------------
        self.log_dict = self.reduce_loss_dict(
            OrderedDict(
                l_kd_feat=l_kd_feat,
                l_kd_out=l_kd_out,
                l_pix=l_pix,
                l_total=l_total,
            )
        )

        # Cache first-sample visuals for TensorBoard (detached, no grad)
        self._vis_lq      = self.lq[:1].detach()
        self._vis_student = student_out[:1].detach()
        self._vis_teacher = pseudo_gt[:1].detach()

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
