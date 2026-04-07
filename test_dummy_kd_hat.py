"""
test_dummy_kd_hat.py
====================
Minimal sanity-check for the train_KD_HAT_S_x3_from_HAT_GAN_x4 pipeline.

Exercises:
  - KDSRModel construction with HAT-type student AND teacher (no RepSR)
  - Student hook resolves to conv_before_upsample (not conv_body)
  - Cross-scale KD: teacher x4 pseudo_GT -> bicubic downsample -> x3 size
  - feed_data + optimize_parameters runs without error
  - All expected loss keys are present and finite
  - Student output shape is correct (B, 3, H*3, W*3)

Uses tiny random-init HAT models (embed_dim=32, depths=[2,2]) on CPU.
No pretrained weights are loaded.
"""

import sys
import shutil
import tempfile
from pathlib import Path

# ── Make hat package importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    from basicsr_compat import apply
    apply()
except Exception:
    pass

import torch

import hat.archs   # registers HAT, RepSR, MambaIRv2 ...
import hat.data    # registers SingleLRDataset ...
import hat.models  # registers KDSRModel ...

from basicsr.utils.registry import MODEL_REGISTRY

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = '[PASS]'
FAIL = '[FAIL]'


def check(label, cond, detail=''):
    status = PASS if cond else FAIL
    msg = f'  {status}  {label}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    if not cond:
        raise AssertionError(f'Check failed: {label}')


def make_opt(tmpdir: str) -> dict:
    """Build a minimal opt dict that mirrors train_KD_HAT_S_x3_from_HAT_GAN_x4.yml
    but uses tiny models so the CPU test finishes in seconds."""
    return {
        # ── top-level ────────────────────────────────────────────────────────
        'name':        'test_KD_HAT_dummy',
        'model_type':  'KDSRModel',
        'scale':       3,        # student target scale
        'num_gpu':     0,        # 0 -> CPU
        'manual_seed': 0,
        'is_train':    True,
        'dist':        False,
        'rank':        0,
        'world_size':  1,
        'find_unused_parameters': False,
        'auto_resume': False,

        # ── paths ────────────────────────────────────────────────────────────
        'path': {
            'pretrain_network_g':       None,   # random init
            'pretrain_network_teacher': None,   # random init
            'strict_load_g':            False,
            'resume_state':             None,
            'experiments_root':         tmpdir,
            'models':                   tmpdir,
            'training_states':          tmpdir,
            'log':                      tmpdir,
            'visualization':            tmpdir,
        },

        # ── student: tiny HAT x3 ─────────────────────────────────────────────
        # (mirrors HAT-S spec but with minimal depth/embed_dim for speed)
        # conv_before_upsample -> Linear(embed_dim=32, num_feat=64) -> 64ch out
        'network_g': {
            'type':          'HAT',
            'upscale':       3,
            'in_chans':      3,
            'img_size':      64,
            'window_size':   8,      # 64 % 8 == 0  -> no extra padding
            'compress_ratio': 3,
            'squeeze_factor': 4,     # embed_dim(32) // 4 = 8 > 0
            'conv_scale':    0.01,
            'overlap_ratio': 0.5,
            'img_range':     1.,
            'depths':        [2, 2], # 2 stages x 2 blocks = very fast on CPU
            'embed_dim':     32,
            'num_heads':     [2, 2], # 32 / 2 = 16 ch/head
            'mlp_ratio':     2,
            'upsampler':     'pixelshuffle',
            'resi_connection': '1conv',
        },

        # ── teacher: tiny HAT x4 ─────────────────────────────────────────────
        # (mirrors HAT-Base/GAN spec but minimal)
        # conv_before_upsample always outputs 64ch regardless of embed_dim
        'network_teacher': {
            'type':          'HAT',
            'upscale':       4,      # <- cross-scale: 4 > 3 (student)
            'in_chans':      3,
            'img_size':      64,
            'window_size':   8,
            'compress_ratio': 3,
            'squeeze_factor': 4,
            'conv_scale':    0.01,
            'overlap_ratio': 0.5,
            'img_range':     1.,
            'depths':        [2, 2],
            'embed_dim':     32,
            'num_heads':     [2, 2],
            'mlp_ratio':     2,
            'upsampler':     'pixelshuffle',
            'resi_connection': '1conv',
        },

        # ── training ─────────────────────────────────────────────────────────
        'train': {
            'ema_decay': 0,
            'optim_g': {
                'type':         'Adam',
                'lr':           2e-4,
                'weight_decay': 0,
                'betas':        [0.9, 0.99],
            },
            'scheduler': {
                'type':       'MultiStepLR',
                'milestones': [100000],
                'gamma':      0.5,
            },
            'total_iter':  2,
            'warmup_iter': -1,

            # Feature KD: both models use conv_before_upsample -> 64ch
            'kd_feat_opt':   {'loss_weight': 0.1},
            'kd_output_opt': {'loss_weight': 1.0},
            'student_feat_channels': 64,  # HAT num_feat is always 64
            'teacher_feat_channels': 64,
        },
    }


# ── Main test ────────────────────────────────────────────────────────────────

def main():
    print()
    print('=' * 60)
    print(' KD HAT-S x3 <- HAT-Base x4  |  Dummy Pipeline Test')
    print('=' * 60)

    tmpdir = tempfile.mkdtemp(prefix='kd_hat_test_')
    try:
        opt = make_opt(tmpdir)

        # ── [1] Model construction ───────────────────────────────────────────
        print('\n[1] Model construction')
        ModelClass = MODEL_REGISTRY.get('KDSRModel')
        model = ModelClass(opt)

        s_params = sum(p.numel() for p in model.net_g.parameters()) / 1e6
        t_params = sum(p.numel() for p in model.net_teacher.parameters()) / 1e6
        check('KDSRModel created', model is not None)
        check('Student on CPU', next(model.net_g.parameters()).device.type == 'cpu')
        check('Teacher on CPU', next(model.net_teacher.parameters()).device.type == 'cpu')
        check('Teacher frozen',
              not any(p.requires_grad for p in model.net_teacher.parameters()))
        print(f'       Student: {s_params:.3f}M params | Teacher: {t_params:.3f}M params')

        # ── [2] Student hook resolves to conv_before_upsample ────────────────
        print('\n[2] Feature hook layers')
        student_inner = getattr(model.net_g, 'module', model.net_g)
        teacher_inner = getattr(model.net_teacher, 'module', model.net_teacher)
        check('Student has conv_before_upsample',
              hasattr(student_inner, 'conv_before_upsample'))
        check('Student does NOT have conv_body (HAT, not RepSR)',
              not hasattr(student_inner, 'conv_body'))
        check('Teacher has conv_before_upsample',
              hasattr(teacher_inner, 'conv_before_upsample'))
        print('       Hook layer: conv_before_upsample -> 64 ch (both)')

        # ── [3] feed_data + 2 optimize_parameters iterations ────────────────
        print('\n[3] Training iterations (cross-scale KD: teacher x4 -> student x3)')
        lq = torch.rand(2, 3, 64, 64)   # batch=2, 64x64 LR patch
        print(f'       LQ  : {tuple(lq.shape)}')
        print(f'       Expected teacher pseudo_GT: (2, 3, 256, 256) -> downsample to (2, 3, 192, 192)')
        print(f'       Expected student output   : (2, 3, 192, 192)')

        for i in range(1, 3):
            model.feed_data({'lq': lq})
            model.optimize_parameters(i)
            log = model.get_current_log()
            loss_str = '  '.join(f'{k}={v:.5f}' for k, v in sorted(log.items()))
            print(f'       iter {i}: {loss_str}')

        # ── [4] Loss keys and finiteness ─────────────────────────────────────
        print('\n[4] Loss validation')
        log = model.get_current_log()
        required_keys = {'l_kd_feat', 'l_kd_out', 'l_total'}
        for key in required_keys:
            check(f'Loss key present: {key}', key in log)
            check(f'Loss finite:      {key}', torch.isfinite(torch.tensor(log[key])).item())

        check('l_total > 0', log['l_total'] > 0)

        # ── [5] Output shape verification ────────────────────────────────────
        print('\n[5] Output shape verification')
        model.net_g.eval()
        with torch.no_grad():
            student_out = model.net_g(lq)
        expected = (2, 3, 192, 192)  # 64 * 3 = 192
        check(f'Student output shape == {expected}',
              tuple(student_out.shape) == expected,
              str(tuple(student_out.shape)))

        # ── [6] Error-case: teacher < student should raise ValueError ─────────
        print('\n[6] Guard: teacher_upscale < student_upscale -> ValueError')
        bad_opt = make_opt(tmpdir)
        bad_opt['network_teacher']['upscale'] = 2   # teacher x2 < student x3
        bad_model = ModelClass(bad_opt)
        bad_model.feed_data({'lq': lq})
        raised = False
        try:
            bad_model.optimize_parameters(1)
        except ValueError as e:
            raised = True
            print(f'       ValueError raised (expected): {e}')
        check('ValueError raised for teacher < student', raised)

        # ── All done ─────────────────────────────────────────────────────────
        print()
        print('=' * 60)
        print(' All checks PASSED')
        print('=' * 60)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
