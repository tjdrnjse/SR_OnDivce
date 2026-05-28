"""
scripts/test_onnx.py
====================
Run inference with a converted ONNX SR model.

Intended for NPU engineers to verify a converted ONNX file produces
correct SR output before compiling it for the target device.

Usage
-----
# Single image
python scripts/test_onnx.py \\
    --onnx   results/onnx/HAT_x3.onnx \\
    --input  datasets/my_lr_images/sample.png \\
    --output results/onnx_test

# Folder of images
python scripts/test_onnx.py \\
    --onnx   results/onnx/HAT_x3.onnx \\
    --input  datasets/my_lr_images \\
    --output results/onnx_test

Notes
-----
- If the input image spatial size does not match the ONNX model's expected
  input size, the image is resized (Lanczos) before inference and the resized
  version is also saved to --output as <stem>_input_resized.png.
- All saved images are PNG.
- Requires: pip install onnxruntime Pillow
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError('onnxruntime is required: pip install onnxruntime')


_IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_session(onnx_path: str) -> ort.InferenceSession:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        sess = ort.InferenceSession(onnx_path,
                                    providers=['CPUExecutionProvider'])
    provider = sess.get_providers()[0]
    print(f'[test_onnx] Provider : {provider}')
    return sess


def get_model_input_shape(sess: ort.InferenceSession):
    """Return (C, H, W) from the first model input."""
    shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 64, 64]
    _, c, h, w = shape
    return int(c), int(h), int(w)


def image_to_tensor(img: Image.Image) -> np.ndarray:
    """PIL RGB image -> float32 numpy (1, 3, H, W) in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = arr.transpose(2, 0, 1)[np.newaxis]         # (1, 3, H, W)
    return arr


def tensor_to_image(arr: np.ndarray) -> Image.Image:
    """float32 numpy (1, 3, H, W) in [0, 1] -> PIL RGB image."""
    arr = arr.squeeze(0).transpose(1, 2, 0)          # (H, W, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def list_images(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.suffix.lower() in _IMG_EXTENSIONS)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_inference(sess: ort.InferenceSession,
                  input_path: Path,
                  output_dir: Path,
                  model_h: int,
                  model_w: int) -> None:
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size        # PIL: (width, height)

    resized = False
    if orig_h != model_h or orig_w != model_w:
        print(f'  {input_path.name}: {orig_w}x{orig_h} -> '
              f'{model_w}x{model_h} (resized)')
        img_input = img.resize((model_w, model_h), Image.LANCZOS)
        resized = True
    else:
        img_input = img

    inp = image_to_tensor(img_input)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: inp})[0]   # (1, 3, H*scale, W*scale)

    sr_img = tensor_to_image(out)

    stem = input_path.stem
    sr_path = output_dir / f'{stem}_sr.png'
    sr_img.save(sr_path)
    print(f'  SR  saved : {sr_path}  [{sr_img.width}x{sr_img.height}]')

    if resized:
        resized_path = output_dir / f'{stem}_input_resized.png'
        img_input.save(resized_path)
        print(f'  Resized input saved : {resized_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Run inference with a converted ONNX SR model.'
    )
    p.add_argument('--onnx', required=True,
                   help='Path to the .onnx model file.')
    p.add_argument('--input', required=True,
                   help='Input image file or folder of images.')
    p.add_argument('--output', default='results/onnx_test',
                   help='Output directory for SR images. '
                        'Default: results/onnx_test')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    onnx_path  = Path(args.onnx)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')
    if not input_path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[test_onnx] ONNX    : {onnx_path}')
    print(f'[test_onnx] Input   : {input_path}')
    print(f'[test_onnx] Output  : {output_dir}')

    sess = load_session(str(onnx_path))
    model_c, model_h, model_w = get_model_input_shape(sess)
    print(f'[test_onnx] Model input shape : (1, {model_c}, {model_h}, {model_w})')

    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f'No images found in: {input_path}')

    print(f'[test_onnx] {len(images)} image(s) to process ...')
    for img_path in images:
        run_inference(sess, img_path, output_dir, model_h, model_w)

    print('[test_onnx] Done.')


if __name__ == '__main__':
    main()
