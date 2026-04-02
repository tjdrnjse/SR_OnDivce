"""
convert_rep_sr.py
=================
Convert a KD-trained RepSR student checkpoint (BasicSR format) into a
mobile-deployment-ready checkpoint where all multi-branch RepSRBlocks
have been fused into single 3×3 convolutions via structural
reparameterization.

Usage
-----
# Basic (auto-detects architecture from the YAML file)
python scripts/convert_rep_sr.py \\
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \\
    --output experiments/converted/RepSR_x4_deployed.pth \\
    --opt    options/train/train_KD_RepSR_x4.yml

# Override architecture parameters manually (no YAML required)
python scripts/convert_rep_sr.py \\
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \\
    --output experiments/converted/RepSR_x4_deployed.pth \\
    --num_feat 64 --num_blocks 8 --upscale 4 \\
    --use_space_to_depth false

Output
------
The saved .pth file contains a plain state-dict (no 'params'/'params_ema'
wrapper) with all RepSRBlock branches merged.  The file can be loaded with:

    import torch
    state = torch.load('RepSR_x4_deployed.pth', map_location='cpu')
    model.load_state_dict(state)
"""

import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict

import torch

# ── Make hat package importable when running from repo root ───────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import hat.archs  # noqa: F401 – triggers @ARCH_REGISTRY.register() for all archs
from hat.archs.rep_sr_arch import RepSR


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Fuse RepSR multi-branch blocks for mobile deployment.'
    )
    p.add_argument('--input', required=True,
                   help='Path to BasicSR-format .pth checkpoint '
                        '(may contain params / params_ema keys).')
    p.add_argument('--output', required=True,
                   help='Output path for the reparameterized .pth file.')
    p.add_argument('--opt', default=None,
                   help='Path to the training YAML to read arch params from.')
    # Manual overrides (used when --opt is not provided)
    p.add_argument('--num_in_ch',  type=int, default=3)
    p.add_argument('--num_feat',   type=int, default=64)
    p.add_argument('--num_blocks', type=int, default=8)
    p.add_argument('--upscale',    type=int, default=4)
    p.add_argument('--use_space_to_depth',
                   type=lambda x: x.lower() not in ('false', '0', 'no'),
                   default=False)
    p.add_argument('--s2d_factor', type=int, default=2)
    p.add_argument('--param_key',  default=None,
                   help='Key to extract from checkpoint dict '
                        '(e.g. "params_ema"). Auto-detected if not set.')
    p.add_argument('--device', default='cpu',
                   help='Device for loading/processing (default: cpu).')
    return p.parse_args()


def load_arch_params_from_yaml(opt_path: str) -> dict:
    """Read network_g section from a BasicSR training YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError('PyYAML is required: pip install pyyaml')

    with open(opt_path, 'r') as f:
        cfg = yaml.safe_load(f)

    ng = cfg.get('network_g', {})
    return {
        'num_in_ch':          ng.get('num_in_ch', 3),
        'num_feat':           ng.get('num_feat', 64),
        'num_blocks':         ng.get('num_blocks', 8),
        'upscale':            ng.get('upscale', 4),
        'use_space_to_depth': ng.get('use_space_to_depth', False),
        's2d_factor':         ng.get('s2d_factor', 2),
    }


def extract_state_dict(ckpt: dict, param_key: str | None) -> OrderedDict:
    """Extract the raw model state-dict from a (possibly wrapped) checkpoint."""
    if param_key is not None:
        if param_key not in ckpt:
            raise KeyError(
                f'Requested param_key "{param_key}" not found in checkpoint. '
                f'Available keys: {list(ckpt.keys())}'
            )
        return ckpt[param_key]

    # Auto-detect
    for key in ('params_ema', 'params', 'state_dict'):
        if key in ckpt:
            print(f'[convert_rep_sr] Auto-detected param key: "{key}"')
            return ckpt[key]

    # Assume the checkpoint itself is a flat state-dict
    print('[convert_rep_sr] No wrapper key found; treating checkpoint as '
          'a flat state-dict.')
    return ckpt


def main():
    args = parse_args()

    # ── 1. Resolve architecture parameters ───────────────────────────────────
    if args.opt:
        print(f'[convert_rep_sr] Reading arch params from: {args.opt}')
        arch_params = load_arch_params_from_yaml(args.opt)
    else:
        arch_params = dict(
            num_in_ch=args.num_in_ch,
            num_feat=args.num_feat,
            num_blocks=args.num_blocks,
            upscale=args.upscale,
            use_space_to_depth=args.use_space_to_depth,
            s2d_factor=args.s2d_factor,
        )

    print('[convert_rep_sr] Architecture parameters:')
    for k, v in arch_params.items():
        print(f'  {k}: {v}')

    # ── 2. Build model (training mode, multi-branch) ──────────────────────────
    model = RepSR(**arch_params)
    model.eval()
    model.to(args.device)

    # ── 3. Load checkpoint ────────────────────────────────────────────────────
    print(f'[convert_rep_sr] Loading checkpoint: {args.input}')
    raw_ckpt = torch.load(args.input, map_location=args.device)

    # If the checkpoint is a plain tensor dict (no nesting)
    if isinstance(raw_ckpt, OrderedDict) or (
        isinstance(raw_ckpt, dict) and
        any(isinstance(v, torch.Tensor) for v in raw_ckpt.values())
    ):
        state_dict = raw_ckpt
    else:
        state_dict = extract_state_dict(raw_ckpt, args.param_key)

    # Strip 'module.' prefix (DistributedDataParallel artefact)
    clean_sd = OrderedDict(
        (k.replace('module.', ''), v) for k, v in state_dict.items()
    )

    missing, unexpected = model.load_state_dict(clean_sd, strict=True)
    if missing:
        print(f'[convert_rep_sr] WARNING – missing keys: {missing}')
    if unexpected:
        print(f'[convert_rep_sr] WARNING – unexpected keys: {unexpected}')

    # ── 4. Structural reparameterization ─────────────────────────────────────
    print('[convert_rep_sr] Reparameterizing RepSR blocks …')
    model.reparameterize()
    print('[convert_rep_sr] Done. All RepSRBlocks fused into single Conv2d.')

    # ── 5. Quick sanity check ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        dummy_in = torch.randn(1, arch_params['num_in_ch'], 64, 64,
                               device=args.device)
        dummy_out = model(dummy_in)
    expected_h = 64 * arch_params['upscale']
    expected_w = 64 * arch_params['upscale']
    assert dummy_out.shape == (1, arch_params['num_in_ch'], expected_h, expected_w), (
        f'Output shape mismatch: {dummy_out.shape}'
    )
    print(f'[convert_rep_sr] Sanity check passed. '
          f'Output shape: {tuple(dummy_out.shape)}')

    # ── 6. Save reparameterized state-dict ────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f'[convert_rep_sr] Saved deployed checkpoint to: {out_path}')

    # Report parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[convert_rep_sr] Model parameters: {n_params / 1e6:.3f} M')


if __name__ == '__main__':
    main()
