"""
scripts/export_onnx.py
======================
Export a teacher SR model (from an inference YAML) to ONNX format.

Intended for mobile/NPU porting feasibility checks.  Only the
``network_t`` (teacher) section of the YAML is used; ``network_g``
(student) keys are ignored.

Usage
-----
# Basic - uses tile_size from YAML as fixed spatial input
python scripts/export_onnx.py --opt options/inference/teacher_hat_x3.yml

# Full control
python scripts/export_onnx.py \\
    --opt       options/inference/teacher_hat_x3.yml \\
    --output-dir results/onnx \\
    --filename  hat_sr_x3.onnx \\
    --opset     17 \\
    --input-h   64 \\
    --input-w   64

Notes
-----
- Input shape is fixed at export time (required for NPU compilers).
  Default spatial size is taken from ``tile_size`` in the YAML (e.g. 256).
  Override with ``--input-h`` / ``--input-w`` if needed.
- Gradient checkpointing (use_checkpoint) is automatically disabled.
- Weights are not required; the script exports with random weights when
  the checkpoint is missing (useful for graph-shape checks).
- MambaIRv2 uses CUDA-only ``mamba_ssm`` ops and cannot be exported to ONNX.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    net_cfg = cfg.get('network_t')
    if net_cfg is None:
        raise ValueError(
            'The YAML must contain a `network_t:` section '
            'with the teacher architecture parameters.'
        )
    net_type = net_cfg['type']
    if 'mamba' in net_type.lower():
        raise RuntimeError(
            f'Model type "{net_type}" relies on CUDA-only mamba_ssm ops '
            'and cannot be exported to ONNX.  '
            'Use the HAT teacher YAML instead.'
        )

    # Apply basicsr torchvision compat shim if present
    try:
        sys.path.insert(0, str(_repo_root))
        import basicsr_compat
        basicsr_compat.apply()
    except ImportError:
        pass

    # Load arch modules directly by file path to avoid hat.__init__
    # (hat.__init__ also imports hat.data which needs basicsr.utils.color_util,
    #  unavailable in older basicsr installations).
    import importlib.util as _ilu
    import os as _os

    # Inject stub hat and hat.archs packages so sub-module imports resolve
    # without triggering hat/__init__.py or hat/archs/__init__.py.
    import types as _types
    for _pkg in ('hat', 'hat.archs'):
        if _pkg not in sys.modules:
            _stub = _types.ModuleType(_pkg)
            _stub.__path__ = [str(_repo_root / _pkg.replace('.', _os.sep))]
            _stub.__package__ = _pkg
            sys.modules[_pkg] = _stub

    arch_folder = _repo_root / 'hat' / 'archs'
    for _fpath in sorted(arch_folder.glob('*_arch.py')):
        _mod_name = f'hat.archs.{_fpath.stem}'
        if _mod_name in sys.modules:
            continue
        _spec = _ilu.spec_from_file_location(_mod_name, _fpath)
        _mod  = _ilu.module_from_spec(_spec)
        sys.modules[_mod_name] = _mod
        _spec.loader.exec_module(_mod)

    from basicsr.utils.registry import ARCH_REGISTRY

    net_cfg = dict(net_cfg)
    net_cfg.pop('type')
    # Disable gradient checkpointing - incompatible with torch.onnx.export
    net_cfg['use_checkpoint'] = False
    return ARCH_REGISTRY.get(net_type)(**net_cfg)


def load_weights(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in ('params_ema', 'params', 'state_dict'):
        if key in ckpt:
            print(f'  [export_onnx] Using param key: "{key}"')
            ckpt = ckpt[key]
            break
    clean = OrderedDict(
        (k.replace('module.', ''), v) for k, v in ckpt.items()
    )
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f'  [export_onnx] WARNING - missing keys ({len(missing)})')
    if unexpected:
        print(f'  [export_onnx] WARNING - unexpected keys ({len(unexpected)})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Export a teacher SR model to ONNX (fixed spatial size).'
    )
    p.add_argument('--opt', required=True,
                   help='Path to inference YAML (options/inference/*.yml).')
    p.add_argument('--output-dir', default='results/onnx',
                   help='Directory to save the .onnx file. Default: results/onnx')
    p.add_argument('--filename', default=None,
                   help='Output filename (e.g. hat_sr_x3.onnx). '
                        'Defaults to <model_type>_x<scale>.onnx')
    p.add_argument('--opset', type=int, default=17,
                   help='ONNX opset version. Default: 17')
    p.add_argument('--input-h', type=int, default=64,
                   help='LR input height in pixels. Must be a multiple of '
                        'window_size (16). Default: 64')
    p.add_argument('--input-w', type=int, default=64,
                   help='LR input width in pixels. Must be a multiple of '
                        'window_size (16). Default: 64')
    p.add_argument('--device', default='cpu',
                   help='Device for tracing (default: cpu).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_yaml(args.opt)

    net_type = cfg.get('network_t', {}).get('type', 'model')
    scale    = cfg.get('scale', cfg.get('network_t', {}).get('upscale', 4))
    in_chans = int(cfg.get('network_t', {}).get('in_chans', 3))

    input_h = args.input_h
    input_w = args.input_w

    window_size = int(cfg.get('network_t', {}).get('window_size', 16))
    if input_h % window_size != 0 or input_w % window_size != 0:
        raise ValueError(
            f'input_h ({input_h}) and input_w ({input_w}) must both be '
            f'multiples of window_size ({window_size}).'
        )

    # ---- Filename / path ----------------------------------------------------
    filename = args.filename or f'{net_type}_x{scale}.onnx'
    if not filename.endswith('.onnx'):
        filename += '.onnx'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    print(f'[export_onnx] Model   : {net_type} x{scale}')
    print(f'[export_onnx] Input   : (1, {in_chans}, {input_h}, {input_w})')
    print(f'[export_onnx] Opset   : {args.opset}')
    print(f'[export_onnx] Output  : {out_path}')
    print(f'[export_onnx] Device  : {args.device}')

    # ---- Build model --------------------------------------------------------
    print('[export_onnx] Building model ...')
    model = build_model(cfg)
    model = model.to(args.device)
    model.eval()

    # ---- Load weights (optional) --------------------------------------------
    ckpt_path = cfg.get('path', {}).get('pretrain_network_t')
    if ckpt_path and Path(ckpt_path).exists():
        print(f'[export_onnx] Loading weights: {ckpt_path}')
        load_weights(model, ckpt_path, args.device)
    else:
        print('[export_onnx] WARNING: checkpoint not found - '
              'exporting with random weights (graph structure only).')

    # ---- Dummy input --------------------------------------------------------
    dummy_input = torch.randn(1, in_chans, input_h, input_w,
                              device=args.device)

    # ---- Verify model runs before export ------------------------------------
    print('[export_onnx] Running forward pass sanity check ...')
    with torch.no_grad():
        out = model(dummy_input)
    expected = (1, in_chans, input_h * scale, input_w * scale)
    assert out.shape == expected, \
        f'Unexpected output shape: {tuple(out.shape)} (expected {expected})'
    print(f'[export_onnx] Forward OK - output shape: {tuple(out.shape)}')

    # ---- ONNX export --------------------------------------------------------
    print(f'[export_onnx] Exporting to ONNX (opset={args.opset}) ...')
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=args.opset,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
    )
    print(f'[export_onnx] Saved: {out_path}')

    # ---- Quick ONNX model check (if onnx is installed) ----------------------
    try:
        import onnx
        model_onnx = onnx.load(str(out_path))
        onnx.checker.check_model(model_onnx)
        print('[export_onnx] onnx.checker.check_model PASSED')
        file_mb = out_path.stat().st_size / 1e6
        print(f'[export_onnx] ONNX file size: {file_mb:.1f} MB')
    except ImportError:
        print('[export_onnx] onnx not installed - skipping model check. '
              '(pip install onnx to enable)')

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[export_onnx] Parameters: {n_params:.3f} M')
    print('[export_onnx] Done.')


if __name__ == '__main__':
    main()
