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

YAML keys for tiling (under root level, same as HATModel):
  tile:
    tile_size: 256
    tile_pad: 32
"""

import math
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


def _tile_forward(model, img, tile_size, tile_pad, scale):
    """Tile an image, run model on each tile, and stitch output.

    Args:
        model (nn.Module): Forward-callable model.
        img (Tensor): (B, C, H, W) input tensor (already padded if needed).
        tile_size (int): Tile spatial size.
        tile_pad (int): Overlap padding around each tile.
        scale (int): SR upscale factor.

    Returns:
        Tensor: (B, C, H*scale, W*scale) stitched output.
    """
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output = img.new_zeros((batch, channel, output_height, output_width))

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[
                :, :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad
            ]

            with torch.no_grad():
                output_tile = model(input_tile)

            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[
                    :, :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile
                ]

    return output


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
        self._teacher_feat: torch.Tensor = None
        self._student_feat: torch.Tensor = None
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

        logger.info(
            f'KDSRModel | teacher: {teacher_opt["type"]} | '
            f'student: {opt["network_g"]["type"]} | '
            f'feat_weight={self.kd_feat_weight} | '
            f'output_weight={self.kd_output_weight}'
        )

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

        For RepSR the feature hook point is `conv_body` which outputs
        (B, num_feat, H, W) (or H/s2d, W/s2d when S2D is enabled).
        """
        student = self.net_g
        if hasattr(student, 'module'):
            student = student.module

        for attr in ('conv_body',):
            layer = getattr(student, attr, None)
            if layer is not None:
                return layer

        raise AttributeError(
            'Cannot find `conv_body` in student model. '
            'Please ensure the student architecture exposes this attribute.'
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

                pseudo_gt = _tile_forward(
                    self.net_teacher,
                    lq_pad,
                    self.opt['tile']['tile_size'],
                    self.opt['tile']['tile_pad'],
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
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt['optim_g']
        )
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # ---- Teacher: generate Pseudo-GT and capture teacher features -------
        # We need teacher features at training resolution (small patches),
        # so we run the teacher on lq directly (no tiling needed for patches).
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
            # Crop padding from pseudo_gt
            _, _, h_out, w_out = pseudo_gt_padded.shape
            scale = self.opt['scale']
            pseudo_gt = pseudo_gt_padded[
                :, :,
                :h_out - pad_h * scale,
                :w_out - pad_w * scale
            ]
            # teacher_feat is stored in self._teacher_feat by hook

        teacher_feat = self._teacher_feat.detach()

        # ---- Student: forward pass, capture student features ----------------
        self.net_g.train()
        student_out = self.net_g(self.lq)
        student_feat = self._student_feat  # (B, student_ch, H', W')

        # ---- Feature-level KD loss (FitNet MSE) ----------------------------
        # Project student features to teacher channel dim
        student_feat_proj = self.feat_projector(student_feat)

        # Align spatial resolution (may differ when S2D is enabled)
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

        Uses parameters from opt['tile']['tile_size'] and
        opt['tile']['tile_pad'].
        """
        model = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        model.eval()
        self.output = _tile_forward(
            model,
            self.img,
            self.opt['tile']['tile_size'],
            self.opt['tile']['tile_pad'],
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
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], img_name,
                        f'{img_name}_{current_iter}.png'
                    )
                else:
                    suffix = self.opt['val'].get('suffix', '')
                    if suffix:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{suffix}.png'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png'
                        )
                imwrite(sr_img, save_img_path)

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
