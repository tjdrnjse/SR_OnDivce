import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class RealHATMSEModel(SRModel):
    """MSE-based Real_HAT Model.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealHATMSEModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @staticmethod
    def _collect_tensor(val, indices=None):
        """Extract a stacked tensor from either a batched Tensor or a list-with-Nones.

        joint_collate_fn may produce a list where some entries are None (missing key
        for that sample). This helper gracefully handles both cases.

        Args:
            val: torch.Tensor (standard collation) or list (joint_collate_fn output)
            indices: optional 1-D LongTensor or list of int to select rows

        Returns:
            torch.Tensor with selected rows stacked on dim-0
        """
        if isinstance(val, torch.Tensor):
            if indices is None:
                return val
            return val[indices]
        # list path — filter out None entries at the requested indices
        idx_list = indices.tolist() if isinstance(indices, torch.Tensor) else (
            list(range(len(val))) if indices is None else list(indices)
        )
        items = [val[i] for i in idx_list if val[i] is not None]
        return torch.stack(items)

    @torch.no_grad()
    def _run_degradation(self, gt, kernel1, kernel2, sinc_kernel):
        """Run the two-stage RealESRGAN degradation pipeline on a GT batch.

        Args:
            gt: (B, C, H, W) GPU tensor — already USM-sharpened if requested
            kernel1, kernel2, sinc_kernel: (B, k, k) GPU tensors

        Returns:
            lq: (B, C, H//scale, W//scale) GPU tensor, clamped & rounded
        """
        ori_h, ori_w = gt.size()[2:4]

        # ---- First degradation ----
        out = filter2D(gt, kernel1)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob, clip=True, rounds=False)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ---- Second degradation ----
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
            mode=mode)
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob, clip=True, rounds=False)

        # JPEG + sinc (two orderings)
        if np.random.uniform() < 0.5:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)

        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        return lq

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.

        Supports three data formats:
          1. All-degradation batch: every sample has kernel1/kernel2/sinc_kernel; use_paired_lq=0.
          2. All-bypass batch: every sample has 'lq'; use_paired_lq=1.  GPU synthesis is skipped.
          3. Mixed batch: some samples are bypass (use_paired_lq=1), others are degradation
             (use_paired_lq=0). Produced when prob_paired_lq in (0, 1) with standard DataLoader.
          4. Legacy paired batch (high_order_degradation=False): has 'lq' and optionally 'gt'.
        """
        use_paired = data.get('use_paired_lq', None)

        # ── Case 1: All-bypass — every sample took the real-LQ path ─────────────
        if (use_paired is not None
                and isinstance(use_paired, torch.Tensor)
                and use_paired.all()):
            self.gt = data['gt'].to(self.device)
            if self.opt.get('gt_usm', True):
                self.gt = self.usm_sharpener(self.gt)
            self.lq = data['lq'].to(self.device)
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()
            return

        if self.is_train and self.opt.get('high_order_degradation', True):

            # ── Case 2: Mixed batch — split by bypass_mask, process separately ──
            if use_paired is not None and isinstance(use_paired, torch.Tensor) and use_paired.any():
                bypass_mask = use_paired.bool()          # True  → real LQ
                degrad_mask = ~bypass_mask               # False → synthesize
                bypass_idx = torch.where(bypass_mask)[0]
                degrad_idx  = torch.where(degrad_mask)[0]

                gt_all = data['gt'].to(self.device)
                if self.opt.get('gt_usm', True):
                    gt_all = self.usm_sharpener(gt_all)

                gt_size   = self.opt['gt_size']
                lq_size   = gt_size // self.opt['scale']
                B, C, H, W = gt_all.shape

                final_gt = torch.empty(B, C, gt_size, gt_size, device=self.device)
                final_lq = torch.empty(B, C, lq_size,  lq_size,  device=self.device)

                # --- bypass sub-batch ---
                if bypass_idx.numel() > 0:
                    gt_bp = gt_all[bypass_idx]
                    lq_bp = self._collect_tensor(data['lq'], bypass_idx).to(self.device)
                    gt_bp, lq_bp = paired_random_crop(gt_bp, lq_bp, gt_size, self.opt['scale'])
                    final_gt[bypass_idx] = gt_bp
                    final_lq[bypass_idx] = lq_bp

                # --- degradation sub-batch ---
                if degrad_idx.numel() > 0:
                    gt_dg = gt_all[degrad_idx]
                    k1   = self._collect_tensor(data['kernel1'],    degrad_idx).to(self.device)
                    k2   = self._collect_tensor(data['kernel2'],    degrad_idx).to(self.device)
                    sinc = self._collect_tensor(data['sinc_kernel'], degrad_idx).to(self.device)
                    lq_dg = self._run_degradation(gt_dg, k1, k2, sinc)
                    gt_dg, lq_dg = paired_random_crop(gt_dg, lq_dg, gt_size, self.opt['scale'])
                    final_gt[degrad_idx] = gt_dg
                    final_lq[degrad_idx] = lq_dg

                self.gt = final_gt
                self.lq = final_lq
                self._dequeue_and_enqueue()
                self.lq = self.lq.contiguous()
                return

            # ── Case 3: All-degradation batch ────────────────────────────────────
            self.gt = data['gt'].to(self.device)
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            kernel1     = self._collect_tensor(data['kernel1']).to(self.device)
            kernel2     = self._collect_tensor(data['kernel2']).to(self.device)
            sinc_kernel = self._collect_tensor(data['sinc_kernel']).to(self.device)

            self.lq = self._run_degradation(self.gt, kernel1, kernel2, sinc_kernel)

            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()

        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealHATMSEModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
