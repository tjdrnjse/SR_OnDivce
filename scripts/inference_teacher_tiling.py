"""
scripts/inference_teacher_tiling.py
====================================
Standalone Teacher SR inference with crop-and-paste tiling.

Supports any Teacher network registered in hat.archs (HAT, MambaIRv2, etc.).
Reads a lightweight YAML config -- only ``network_t`` (teacher architecture)
and ``path.pretrain_network_t`` (teacher weights) are required.
All ``network_g`` / student keys are silently ignored.

Tiling strategy: crop-and-paste
---------------------------------
Each HR output tile's border is cropped by ``(tile_overlap * scale) // 2``
pixels per side (skipping outer-image edges).  Only the safe central region is
pasted into the final HR canvas.  This eliminates boundary artifacts without
requiring gradient-based blending.

Steps
-----
1. Load LR image at native resolution (no resize).
2. Reflection-pad the image so its dimensions fit the tile grid
   (stride = tile_size - tile_overlap).
3. Optionally pad to a multiple of ``window_size`` for transformer teachers.
4. For each tile: run teacher model, crop borders, paste safe region.
5. Remove the padding region from the HR output.
6. Save as PNG.

Usage
-----
python scripts/inference_teacher_tiling.py \\
    --opt   options/inference/teacher_hat_x3.yml \\
    --input datasets/my_lr_images \\
    --output results/teacher_hat_x3

Optional flags
--------------
--device  cuda | cpu     Override device (default: auto-detect)
--suffix  _hat_x3        Suffix appended to each saved filename stem
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Make hat package importable when running from the repo root
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import hat.archs  # noqa: F401 -- triggers @ARCH_REGISTRY.register() for all archs
from basicsr.utils.registry import ARCH_REGISTRY


# ---------------------------------------------------------------------------
# YAML / CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Teacher SR tiling inference (standalone, no student needed).'
    )
    p.add_argument('--opt',    required=True,
                   help='Path to inference YAML (see options/inference/ for examples).')
    p.add_argument('--input',  default=None,
                   help='LR image folder. Overrides YAML input_dir.')
    p.add_argument('--output', default=None,
                   help='Output folder for SR images. Overrides YAML output_dir.')
    p.add_argument('--device', default=None,
                   help='Compute device (e.g. "cuda", "cuda:1", "cpu"). '
                        'Overrides YAML device. Default: auto-detect CUDA.')
    p.add_argument('--suffix', default='',
                   help='Suffix appended to each output filename stem '
                        '(e.g. "_hat_x3"). Default: "_SR".')
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_teacher(cfg: dict) -> torch.nn.Module:
    """Instantiate the teacher model from ``cfg['network_t']``."""
    net_cfg = cfg.get('network_t')
    if net_cfg is None:
        raise ValueError(
            'The inference YAML must contain a `network_t:` section '
            'with the teacher architecture parameters.'
        )
    # Copy so we don't mutate the caller's dict when we pop 'type'
    net_cfg = dict(net_cfg)
    net_type = net_cfg.pop('type')
    return ARCH_REGISTRY.get(net_type)(**net_cfg)


def load_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a BasicSR-format checkpoint (auto-detects param key wrapper)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Unwrap common BasicSR checkpoint wrappers
    for key in ('params_ema', 'params', 'state_dict'):
        if key in ckpt:
            print(f'  [load_weights] Using param key: "{key}"')
            ckpt = ckpt[key]
            break
    # Strip DataParallel / DistributedDataParallel prefix
    clean = OrderedDict(
        (k.replace('module.', ''), v) for k, v in ckpt.items()
    )
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f'  [load_weights] WARNING - missing keys ({len(missing)}): '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  [load_weights] WARNING - unexpected keys ({len(unexpected)}): '
              f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(folder: str) -> List[Path]:
    """Return sorted list of image paths in *folder*."""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )


def read_image_bgr(path: Path) -> np.ndarray:
    """Read image as float32 in [0, 1], shape (H, W, C) BGR."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'Cannot read image: {path}')
    return img.astype(np.float32) / 255.0


def img_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, C) BGR float32 -> (1, C, H, W) RGB float32 tensor on *device*."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return t.to(device)


def tensor_to_img_bgr(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) or (C, H, W) tensor -> (H, W, C) uint8 BGR numpy array."""
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.clamp(0.0, 1.0).float().cpu().numpy()  # (C, H, W)
    arr = arr.transpose(1, 2, 0)                    # (H, W, C) RGB
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return (arr * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _pad_len(size: int, tile_size: int, stride: int) -> int:
    """Pixels to pad so the tile grid covers *size* exactly.

    After padding, ``(size + pad - tile_size)`` is divisible by *stride*,
    and at least one full tile fits.
    """
    if size <= tile_size:
        return tile_size - size
    remainder = (size - tile_size) % stride
    return (stride - remainder) % stride


def _tile_starts(padded_size: int, tile_size: int, stride: int) -> List[int]:
    """Return LR start positions of tiles along one axis."""
    starts = list(range(0, padded_size - tile_size + 1, stride))
    # Safety: ensure last tile covers the trailing edge
    if starts[-1] + tile_size < padded_size:
        starts.append(padded_size - tile_size)
    return starts


# ---------------------------------------------------------------------------
# Core tiling inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def tiled_sr(model: torch.nn.Module,
             lq: torch.Tensor,
             tile_size: int,
             tile_overlap: int,
             upscale: int,
             window_size: int = 1) -> torch.Tensor:
    """Run SR inference with crop-and-paste tiling.

    Args:
        model:        Teacher SR model in eval mode on the correct device.
        lq:           (1, C, H, W) LR input tensor in [0, 1].
        tile_size:    LR tile spatial size (e.g. 256).
        tile_overlap: LR-space overlap between adjacent tiles (e.g. 32).
        upscale:      SR scale factor.
        window_size:  Transformer window size for window-aligned input padding.
                      Pass 1 (default) to disable extra alignment padding.

    Returns:
        (1, C, H*upscale, W*upscale) HR output tensor.
    """
    _, C, H, W = lq.shape
    stride = tile_size - tile_overlap

    # Number of HR border pixels to crop from each tile side
    # (tile_overlap * upscale) / 2 ensures adjacent safe regions abut exactly
    c = (tile_overlap * upscale) // 2

    # ---- 1. Pad LR input to align with tile grid ---------------------------
    pad_h = _pad_len(H, tile_size, stride)
    pad_w = _pad_len(W, tile_size, stride)

    # Additional alignment to window_size (required for HAT / MambaIRv2)
    H_tmp = H + pad_h
    W_tmp = W + pad_w
    wpad_h = (window_size - H_tmp % window_size) % window_size
    wpad_w = (window_size - W_tmp % window_size) % window_size

    total_h = pad_h + wpad_h
    total_w = pad_w + wpad_w

    if total_h > 0 or total_w > 0:
        # Reflection padding fails when pad >= image dim; fall back to replicate
        pad_mode = 'reflect' if total_h < H and total_w < W else 'replicate'
        lq_pad = F.pad(lq, (0, total_w, 0, total_h), mode=pad_mode)
    else:
        lq_pad = lq

    _, _, H_pad, W_pad = lq_pad.shape

    # ---- 2. Compute tile positions (LR space) --------------------------------
    ys = _tile_starts(H_pad, tile_size, stride)
    xs = _tile_starts(W_pad, tile_size, stride)

    # ---- 3. Allocate HR output canvas ----------------------------------------
    out_H = H_pad * upscale
    out_W = W_pad * upscale
    output = lq.new_zeros(1, C, out_H, out_W)

    # ---- 4. Process each tile (crop borders, paste safe region) --------------
    for i, yr in enumerate(ys):
        for j, xr in enumerate(xs):
            tile_lr = lq_pad[:, :, yr:yr + tile_size, xr:xr + tile_size]
            tile_hr = model(tile_lr)   # (1, C, tile_size*upscale, tile_size*upscale)

            th = tile_hr.shape[2]
            tw = tile_hr.shape[3]

            # Do NOT crop the outer edges of the very first / last tile
            crop_top    = c if i > 0 else 0
            crop_left   = c if j > 0 else 0
            crop_bottom = c if i < len(ys) - 1 else 0
            crop_right  = c if j < len(xs) - 1 else 0

            bot_idx = th - crop_bottom if crop_bottom else th
            rgt_idx = tw - crop_right  if crop_right  else tw

            safe = tile_hr[:, :, crop_top:bot_idx, crop_left:rgt_idx]

            paste_y = yr * upscale + crop_top
            paste_x = xr * upscale + crop_left
            sh, sw = safe.shape[2], safe.shape[3]
            output[:, :, paste_y:paste_y + sh, paste_x:paste_x + sw] = safe

    # ---- 5. Crop back to original HR resolution (remove padding) -------------
    return output[:, :, :H * upscale, :W * upscale]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_yaml(args.opt)

    # Resolve settings (CLI overrides YAML)
    input_dir  = args.input  or cfg.get('input_dir',  '')
    output_dir = args.output or cfg.get('output_dir', 'results/teacher_inference')
    device_str = args.device or cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device     = torch.device(device_str)
    suffix     = args.suffix if args.suffix else '_SR'

    tile_size    = int(cfg.get('tile_size',    256))
    tile_overlap = int(cfg.get('tile_overlap', 32))
    upscale      = int(cfg.get('scale',        4))

    if not input_dir:
        raise ValueError(
            'No input directory specified. '
            'Set input_dir in the YAML or pass --input <path>.'
        )

    # ---- Build teacher model ------------------------------------------------
    print('[inference_teacher_tiling] Building teacher model ...')
    model = build_teacher(cfg)
    model = model.to(device)
    model.eval()

    ckpt_path = cfg.get('path', {}).get('pretrain_network_t')
    if ckpt_path:
        print(f'[inference_teacher_tiling] Loading weights: {ckpt_path}')
        load_weights(model, ckpt_path)
    else:
        print('[inference_teacher_tiling] WARNING: no checkpoint path found '
              '(path.pretrain_network_t). Running with random weights.')

    # Detect transformer window_size for aligned padding
    teacher_inner = getattr(model, 'module', model)
    window_size   = getattr(teacher_inner, 'window_size', 1)
    n_params      = sum(p.numel() for p in model.parameters()) / 1e6

    print(f'[inference_teacher_tiling] Teacher: {cfg["network_t"]["type"]} | '
          f'{n_params:.2f} M params | scale x{upscale} | '
          f'tile={tile_size} overlap={tile_overlap} window={window_size} | '
          f'device={device}')

    # ---- Prepare output directory -------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Process images ------------------------------------------------------
    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f'No images found in: {input_dir}')

    print(f'[inference_teacher_tiling] {len(image_paths)} images to process ...')

    for img_path in image_paths:
        img_bgr = read_image_bgr(img_path)
        H, W    = img_bgr.shape[:2]
        lq      = img_to_tensor(img_bgr, device)

        sr = tiled_sr(
            model, lq,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            upscale=upscale,
            window_size=window_size,
        )

        sr_img = tensor_to_img_bgr(sr)

        out_name = img_path.stem + suffix + '.png'
        save_path = Path(output_dir) / out_name
        cv2.imwrite(str(save_path), sr_img)

        print(f'  {img_path.name:40s}  '
              f'{W}x{H} -> {sr_img.shape[1]}x{sr_img.shape[0]}  '
              f'-> {save_path.name}')

    print('[inference_teacher_tiling] All done.')


if __name__ == '__main__':
    main()
