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

def _make_blend_weights(size: int, overlap: int, device) -> torch.Tensor:
    """Build a 2-D linear-ramp blending weight mask of shape (size, size).

    The central ``size - 2*fade`` pixels have weight 1.0.
    The border ``fade = overlap // 2`` pixels ramp linearly from
    ``1 / (fade + 1)`` (outermost edge) up to 1.0 (inner boundary).
    All values are strictly > 0, so dividing by the accumulated weight map
    never produces division-by-zero.

    Args:
        size (int):    HR-space tile side length (tile_size * upscale).
        overlap (int): HR-space overlap between adjacent tiles
                       (tile_overlap * upscale).
        device:        Target device.

    Returns:
        Tensor: (size, size) float32 weight mask.
    """
    fade = max(1, overlap // 2)
    w = torch.ones(size, dtype=torch.float32, device=device)
    if size > 2 * fade:
        ramp = torch.linspace(1.0 / (fade + 1), float(fade) / (fade + 1),
                              fade, device=device)
        w[:fade]        = ramp           # left / top edge ramps up
        w[size - fade:] = ramp.flip(0)  # right / bottom edge ramps down
    # 2-D mask: outer product of two 1-D ramps  (size, size)
    return w.unsqueeze(0) * w.unsqueeze(1)


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
             window_size: int = 1,
             tile_batch_size: int = 16,
             use_bf16: bool = False) -> torch.Tensor:
    """Run SR inference with batched Linear-Blending (Feathering) tiling.

    Replaces the previous Crop-and-Paste approach with weight-accumulation
    blending to eliminate mosaic/grid artifacts at tile boundaries:

      output_canvas  +=  tile_hr  * blend_mask   (weighted accumulation)
      weight_canvas  +=  blend_mask
      output_final    =  output_canvas / clamp(weight_canvas, min=1e-8)

    The 2-D blend mask linearly ramps from a small positive value at the
    tile edge to 1.0 in the centre, so overlapping regions are merged
    smoothly without visible seams.

    Additional optimisations retained from the previous version:
      - **Tile batching**: tiles are forwarded in mini-batches of
        ``tile_batch_size`` to eliminate GPU starvation.
      - **BF16 autocast** (CUDA only): H100 Tensor Cores are used for the
        model forward; results are cast back to FP32 before accumulation.

    Args:
        model:           Teacher SR model in eval mode on the correct device.
        lq:              (1, C, H, W) LR input tensor in [0, 1], float32.
        tile_size:       LR tile spatial size (e.g. 256).
        tile_overlap:    LR-space overlap between adjacent tiles (e.g. 32).
        upscale:         SR scale factor.
        window_size:     Transformer window size for window-aligned padding.
                         Pass 1 (default) to disable extra alignment padding.
        tile_batch_size: Number of LR tiles forwarded per model call (default 16).

    Returns:
        (1, C, H*upscale, W*upscale) HR output tensor, float32.
    """
    _, C, H, W = lq.shape
    device = lq.device
    stride = tile_size - tile_overlap

    # ---- 1. Pad LR input to fit tile grid ------------------------------------
    # Reflection-pad so (H_pad - tile_size) is divisible by stride,
    # then additionally align to window_size for transformer teachers.
    pad_h = _pad_len(H, tile_size, stride)
    pad_w = _pad_len(W, tile_size, stride)

    H_tmp = H + pad_h
    W_tmp = W + pad_w
    wpad_h = (window_size - H_tmp % window_size) % window_size
    wpad_w = (window_size - W_tmp % window_size) % window_size

    total_h = pad_h + wpad_h
    total_w = pad_w + wpad_w

    if total_h > 0 or total_w > 0:
        pad_mode = 'reflect' if total_h < H and total_w < W else 'replicate'
        lq_pad = F.pad(lq, (0, total_w, 0, total_h), mode=pad_mode)
        # lq_pad: (1, C, H+total_h, W+total_w)  float32
    else:
        lq_pad = lq
        # lq_pad: (1, C, H, W)  float32

    _, _, H_pad, W_pad = lq_pad.shape

    # ---- 2. Compute tile start positions (LR space) --------------------------
    ys = _tile_starts(H_pad, tile_size, stride)   # len = n_rows
    xs = _tile_starts(W_pad, tile_size, stride)   # len = n_cols
    n_total = len(ys) * len(xs)

    # ---- 3. Collect all LR tile views into a flat list -----------------------
    # tile_views: list of (C, tile_size, tile_size) zero-copy views of lq_pad
    # tile_meta:  parallel list of (yr, xr) LR start coordinates
    tile_meta  = []
    tile_views = []
    for yr in ys:
        for xr in xs:
            tile_meta.append((yr, xr))
            tile_views.append(lq_pad[0, :, yr:yr + tile_size, xr:xr + tile_size])
            # Each view: (C, tile_size, tile_size)  float32

    # ---- 4. Allocate FP32 HR accumulation canvases ---------------------------
    hr_tile = tile_size * upscale          # HR tile side length
    out_H   = H_pad * upscale
    out_W   = W_pad * upscale

    output_canvas = lq.new_zeros(1, C, out_H, out_W)
    # output_canvas: (1, C, H_pad*upscale, W_pad*upscale)  float32

    weight_canvas = lq.new_zeros(1, 1, out_H, out_W)
    # weight_canvas: (1, 1, H_pad*upscale, W_pad*upscale)  float32

    # Pre-compute 2-D blend mask (same shape for every tile)
    # blend: (hr_tile, hr_tile) float32
    # blend[i,j] ramps from 1/(fade+1) at edges to 1.0 in the centre
    blend = _make_blend_weights(hr_tile, tile_overlap * upscale, device)
    # Expand for broadcasting with (B, C, hr_tile, hr_tile)
    blend_bc = blend.unsqueeze(0).unsqueeze(0)   # (1, 1, hr_tile, hr_tile)

    # BF16 autocast: enabled only when caller requests it AND device supports it
    use_bf16 = use_bf16 and (device.type == 'cuda')

    # ---- 5. Forward in mini-batches, accumulate with blend mask --------------
    for batch_start in range(0, n_total, tile_batch_size):
        batch_end = min(batch_start + tile_batch_size, n_total)

        # --- 5a. Stack LR tiles into a mini-batch ----------------------------
        # lr_batch: (B, C, tile_size, tile_size)  float32
        #   B = batch_end - batch_start  (≤ tile_batch_size)
        lr_batch = torch.stack(tile_views[batch_start:batch_end])

        # --- 5b. Forward: BF16 on CUDA, FP32 on CPU --------------------------
        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                hr_batch = model(lr_batch)
                # hr_batch: (B, C, hr_tile, hr_tile)  bfloat16
            hr_batch = hr_batch.float()
            # hr_batch: (B, C, hr_tile, hr_tile)  float32
        else:
            hr_batch = model(lr_batch)
            # hr_batch: (B, C, hr_tile, hr_tile)  float32

        # --- 5c. Weighted accumulation into canvases -------------------------
        for k, (yr, xr) in enumerate(tile_meta[batch_start:batch_end]):
            # tile_hr: (C, hr_tile, hr_tile)  float32  — single tile from batch
            tile_hr = hr_batch[k]

            y_hr = yr * upscale
            x_hr = xr * upscale

            # Accumulate: output_canvas += tile_hr * blend_mask
            # (C, hr_tile, hr_tile) * (1, hr_tile, hr_tile) -> broadcasts over C
            output_canvas[0, :, y_hr:y_hr + hr_tile, x_hr:x_hr + hr_tile] += (
                tile_hr * blend
            )

            # Accumulate blend weights (same for every channel)
            weight_canvas[0, 0, y_hr:y_hr + hr_tile, x_hr:x_hr + hr_tile] += blend

    # ---- 6. Normalize: divide pixel sums by accumulated weight sums ----------
    # Clamp prevents division-by-zero (all weights are > 0 by construction,
    # but the clamp adds safety for any uncovered padded region).
    # output_final: (1, C, H_pad*upscale, W_pad*upscale)  float32
    output_final = output_canvas / weight_canvas.clamp(min=1e-8)

    # ---- 7. Crop padding back to original HR resolution ----------------------
    # (1, C, H_pad*upscale, W_pad*upscale) -> (1, C, H*upscale, W*upscale)
    return output_final[:, :, :H * upscale, :W * upscale]


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

    tile_size       = int(cfg.get('tile_size',       256))
    tile_overlap    = int(cfg.get('tile_overlap',    32))
    tile_batch_size = int(cfg.get('tile_batch_size', 16))
    upscale         = int(cfg.get('scale',           4))
    use_bf16        = bool(cfg.get('use_bf16',       False))

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

    bf16_status = 'on' if (use_bf16 and device.type == 'cuda') else 'off'
    print(f'[inference_teacher_tiling] Teacher: {cfg["network_t"]["type"]} | '
          f'{n_params:.2f} M params | scale x{upscale} | '
          f'tile={tile_size} overlap={tile_overlap} batch={tile_batch_size} '
          f'window={window_size} | device={device} | bf16={bf16_status}')

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
            tile_batch_size=tile_batch_size,
            use_bf16=use_bf16,
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
