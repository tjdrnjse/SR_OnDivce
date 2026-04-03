"""
Single LR Dataset — loads real LR images as-is (no resize / downsampling).

Designed for the KD-SR pipeline where the teacher generates pseudo-GT
on-the-fly, so only LR images are required on disk.

YAML example::

    datasets:
      train:
        name: LR_train
        type: SingleLRDataset
        dataroot_lq: datasets/your_lr_images
        io_backend:
          type: disk
        lq_patch_size: 64
        use_hflip: true
        use_rot: true
"""

import cv2
import os.path as osp
import random

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize

try:
    from basicsr.data.data_util import scandir
except ImportError:
    from basicsr.utils import scandir


@DATASET_REGISTRY.register()
class SingleLRDataset(data.Dataset):
    """Dataset that loads LR images without any resize or downsampling.

    Only a random crop to ``lq_patch_size`` is applied.  No bicubic or
    other downsampling is performed.  The KD teacher model receives this
    LR patch and produces the Pseudo-GT HR patch on-the-fly.

    Args:
        opt (dict): Options containing the keys below.

    Required keys:
        dataroot_lq (str): Root directory of LR images.
        io_backend (dict): IO backend config (e.g. ``{type: disk}``).
        lq_patch_size (int): Spatial size of the random crop (LR space).

    Optional keys:
        meta_info_file (str): Path to a text file listing relative image
            paths (one per line).  If absent, all images under
            ``dataroot_lq`` are used.
        use_hflip (bool): Apply random horizontal flip.  Default: False.
        use_rot (bool): Apply random 90-degree rotations.  Default: False.
        mean / std (list): Per-channel normalisation values.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['dataroot_lq']
        self.lq_patch_size = opt.get('lq_patch_size', 64)

        # Build file list
        if 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file']) as f:
                self.paths = [
                    osp.join(self.lq_folder, line.strip().split()[0])
                    for line in f if line.strip()
                ]
        else:
            self.paths = sorted(
                list(scandir(self.lq_folder, full_path=True))
            )

        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)  # HWC, BGR, [0,1]

        patch = self.lq_patch_size

        # --- Pad if image is smaller than the patch size ------------------
        h, w, _ = img_lq.shape
        if h < patch or w < patch:
            pad_h = max(0, patch - h)
            pad_w = max(0, patch - w)
            img_lq = cv2.copyMakeBorder(
                img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
            h, w, _ = img_lq.shape

        # --- Random crop (no resize) -------------------------------------
        top = random.randint(0, h - patch)
        left = random.randint(0, w - patch)
        img_lq = img_lq[top:top + patch, left:left + patch, :]

        # --- Augmentation ------------------------------------------------
        img_lq = augment(
            img_lq,
            self.opt.get('use_hflip', False),
            self.opt.get('use_rot', False),
        )

        # --- BGR -> RGB, HWC -> CHW, numpy -> tensor ---------------------
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
