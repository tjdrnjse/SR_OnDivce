import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from basicsr.data.data_util import scandir

@DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        # ── Step 1: Paired LQ bypass ──────────────────────────────────────────
        # If `dataroot_lq` is given in the YAML and `prob_paired_lq` > 0,
        # a fraction of __getitem__ calls will bypass on-the-fly degradation
        # and load a real paired LQ image directly (supervised learning).
        # When the probability fires, GT and LQ are jointly augmented/cropped
        # so that they remain spatially aligned.
        #
        # Required YAML keys (in dataset section):
        #   dataroot_lq    : path to folder containing real paired LQ images
        #   prob_paired_lq : float in [0, 1], default 0.0
        #   scale          : SR upscale factor (needed for synchronized crop)
        self.lq_folder = opt.get('dataroot_lq', None)
        self.prob_paired_lq = float(opt.get('prob_paired_lq', 0.0))
        self.scale = opt.get('scale', 4)   # used to compute LQ pad size
        self.paths_lq = None

        if self.lq_folder is not None and self.prob_paired_lq > 0:
            self.paths_lq = sorted(list(scandir(self.lq_folder, full_path=True)))
            if not self.paths_lq:
                raise ValueError(
                    f'RealESRGANDataset: No LQ images found in '
                    f'dataroot_lq: {self.lq_folder}')
            logger = get_root_logger()
            logger.info(
                f'RealESRGANDataset: probabilistic paired-LQ bypass enabled. '
                f'GT={len(self.paths)} / LQ={len(self.paths_lq)} images, '
                f'prob_paired_lq={self.prob_paired_lq}')
        elif self.lq_folder is not None and self.prob_paired_lq == 0:
            logger = get_root_logger()
            logger.warning(
                'RealESRGANDataset: dataroot_lq is set but prob_paired_lq=0. '
                'Paired LQ bypass is effectively disabled.')

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # ── Step 2a: Decide bypass vs. degradation BEFORE augmentation ────────
        # Bypass fires when dataroot_lq is configured AND the random draw wins.
        use_bypass = (
            self.paths_lq is not None
            and random.random() < self.prob_paired_lq
        )
        img_lq = None
        lq_path = None

        if use_bypass:
            # Load the paired LQ image (index is cycled if datasets differ in size)
            lq_idx = index % len(self.paths_lq)
            lq_path = self.paths_lq[lq_idx]
            try:
                lq_bytes = self.file_client.get(lq_path, 'lq')
                img_lq = imfrombytes(lq_bytes, float32=True)
            except (IOError, OSError):
                # Fall back to degradation path if the paired LQ cannot be loaded
                use_bypass = False
                img_lq = None

        # ── Step 2b: Augmentation — apply SAME transforms to GT (and LQ) ─────
        # augment() accepts a list; both images receive identical flip/rotation.
        if use_bypass and img_lq is not None:
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt['use_hflip'],
                self.opt.get('use_rot', False)
            )
        else:
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # ── Step 2c: Pad/crop GT to crop_pad_size — track coordinates ────────
        # gt_top / gt_left are used to compute the synchronized LQ crop below.
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        gt_top, gt_left = 0, 0

        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            gt_top = random.randint(0, h - crop_pad_size)
            gt_left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[gt_top:gt_top + crop_pad_size, gt_left:gt_left + crop_pad_size, ...]

        # ── Step 2d: Paired LQ bypass — synchronized crop and early return ────
        if use_bypass and img_lq is not None:
            # LQ target pad size (mirrors GT pad size at LR resolution)
            lq_pad = crop_pad_size // self.scale  # e.g. 400 // 4 = 100

            h_lq, w_lq = img_lq.shape[:2]

            # Pad LQ to lq_pad if smaller
            if h_lq < lq_pad or w_lq < lq_pad:
                img_lq = cv2.copyMakeBorder(
                    img_lq,
                    0, max(0, lq_pad - h_lq),
                    0, max(0, lq_pad - w_lq),
                    cv2.BORDER_REFLECT_101
                )
                h_lq, w_lq = img_lq.shape[:2]

            # Synchronized crop: GT crop coords scaled down to LR space.
            # Clamped so the slice never exceeds the LQ image boundary.
            lq_top  = min(gt_top  // self.scale, max(0, h_lq - lq_pad))
            lq_left = min(gt_left // self.scale, max(0, w_lq - lq_pad))
            img_lq = img_lq[lq_top:lq_top + lq_pad, lq_left:lq_left + lq_pad, ...]

            # BGR → RGB, HWC → CHW
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

            return {
                'gt': img_gt,
                'lq': img_lq,
                'gt_path': gt_path,
                'lq_path': lq_path,
                'use_paired_lq': torch.tensor(1, dtype=torch.int),
            }

        # ── Degradation path (unchanged) — falls through to kernel generation ─

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            'gt': img_gt,
            'kernel1': kernel,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path,
            # Step 2: flag so feed_data can distinguish this path from bypass
            'use_paired_lq': torch.tensor(0, dtype=torch.int),
        }
        return return_d

    def __len__(self):
        return len(self.paths)