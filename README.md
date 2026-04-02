# HAT + Knowledge Distillation SR Pipeline

A **turn-key Knowledge Distillation (KD) Super-Resolution** framework built on top of [HAT](https://github.com/XPixelGroup/HAT) (BasicSR).

- **Teacher**: HAT *or* MambaIRv2 – large, frozen transformer model
- **Student**: RepSR – lightweight RepVGG-based model with structural reparameterization
- **KD method**: FitNet (feature-level MSE) + output-level L1 against teacher pseudo-GT
- **Extras**: optional Space-to-Depth pre-processing, tiling for GPU-memory-limited inference

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset & Directory Layout](#2-dataset--directory-layout)
3. [YAML Option Reference](#3-yaml-option-reference)
4. [KD Training](#4-kd-training)
5. [Convert to Deployment Weights](#5-convert-to-deployment-weights)
6. [Final Inference / Test](#6-final-inference--test)
7. [Architecture Notes](#7-architecture-notes)

---

## 1. Environment Setup

```bash
# 1-a. Clone & install HAT dependencies
git clone https://github.com/XPixelGroup/HAT.git
cd HAT
pip install -r requirements.txt
pip install basicsr

# 1-b. Install HAT as editable package
python setup.py develop

# 1-c. (Optional) MambaIRv2 teacher requires extra packages
pip install mamba-ssm einops
# Note: mamba-ssm needs a CUDA-capable GPU and matching CUDA toolkit.
# Skip this if you are only using the HAT teacher.
```

---

## 2. Dataset & Directory Layout

### Recommended directory structure

```
HAT/
├── datasets/
│   ├── DF2K/
│   │   ├── DF2K_HR_sub/          # HR sub-images (800x800 -> 480x480 patches)
│   │   └── DF2K_bicx4_sub/       # LR sub-images (bicubic x4 downsampled)
│   ├── Set5/
│   │   ├── GTmod4/               # HR ground-truth
│   │   └── LRbicx4/              # LR bicubic inputs
│   └── Set14/
│       ├── GTmod4/
│       └── LRbicx4/
└── experiments/
    └── pretrained_models/
        └── HAT_SRx4_ImageNet-pretrain.pth   # teacher checkpoint
```

### Generating DF2K sub-images (if not already done)

```bash
# Generate HR sub-images (stride 240, size 480)
python basicsr/scripts/extract_subimages.py \
    --input  datasets/DF2K/DF2K_HR \
    --output datasets/DF2K/DF2K_HR_sub \
    --n_thread 20

# Generate LR sub-images
python basicsr/scripts/extract_subimages.py \
    --input  datasets/DF2K/DF2K_LR_bicubic/X4 \
    --output datasets/DF2K/DF2K_bicx4_sub \
    --n_thread 20
```

Download pretrained teacher checkpoints from the official HAT repo:
https://github.com/XPixelGroup/HAT#pretrained-models

---

## 3. YAML Option Reference

All config files live in `options/train/` and `options/test/`.
The KD pipeline uses `options/train/train_KD_RepSR_x4.yml`.

### 3-a. Top-level settings

| Key | Description | Example |
|-----|-------------|---------|
| `model_type` | Must be `KDSRModel` | `KDSRModel` |
| `scale` | SR upscale factor | `4` |
| `num_gpu` | Number of GPUs (`auto` = all visible) | `auto` |

### 3-b. Student network (`network_g`)

```yaml
network_g:
  type: RepSR
  num_in_ch: 3               # input channels (3 = RGB)
  num_feat: 64               # internal feature channels (controls width)
  num_blocks: 8              # number of RepSR blocks (controls depth)
  upscale: 4                 # SR scale factor
  use_space_to_depth: false  # true -> apply nn.PixelUnshuffle at input
  s2d_factor: 2              # PixelUnshuffle downscale factor (S2D=true only)
```

**Space-to-Depth mode** (`use_space_to_depth: true`)

Input is passed through `nn.PixelUnshuffle(s2d_factor)` before the main body.
The effective upscale inside the network becomes `upscale x s2d_factor`.
This lets the body operate on lower-resolution feature maps, which can
reduce computation while keeping the final output resolution unchanged.

### 3-c. Teacher network (`network_teacher`)

Select HAT or MambaIRv2 by changing `type`:

```yaml
# HAT teacher
network_teacher:
  type: HAT
  upscale: 4
  img_size: 64
  window_size: 16
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  upsampler: 'pixelshuffle'
  # ... (see options/train/train_KD_RepSR_x4.yml for full list)

# MambaIRv2 teacher (requires mamba-ssm)
network_teacher:
  type: MambaIRv2
  upscale: 4
  embed_dim: 60
  d_state: 8
  depths: [6, 6, 6, 6]
  # ... (see commented block in train_KD_RepSR_x4.yml)
```

### 3-d. Loss weights (`train`)

```yaml
train:
  pixel_opt:
    loss_weight: 0.5     # supervised L1 vs real HR GT (set 0 to disable)

  kd_feat_opt:
    loss_weight: 1.0     # FitNet MSE: projected student feat vs teacher feat

  kd_output_opt:
    loss_weight: 1.0     # L1: student output vs teacher pseudo-GT

  student_feat_channels: 64   # must equal network_g.num_feat
  teacher_feat_channels: 64   # fixed: HAT / MambaIRv2 conv_before_upsample = 64
```

### 3-e. Tiling (`tile`)

Enable tiling for GPU-memory-limited validation or inference:

```yaml
tile:
  tile_size: 256   # spatial tile size in pixels (multiple of window_size=16)
  tile_pad: 32     # overlap padding between adjacent tiles (also multiple of 16)
```

Remove (or comment out) the `tile:` block to process images in one pass.

---

## 4. KD Training

```bash
# Single-GPU training
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml

# Multi-GPU training (e.g., 4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=4321 \
    hat/train.py -opt options/train/train_KD_RepSR_x4.yml \
    --launcher pytorch

# Resume from a checkpoint
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml \
    --auto_resume
```

**Key paths to set in the YAML before training:**

```yaml
path:
  pretrain_network_teacher: ./experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth
  pretrain_network_g: ~          # or a student warmup checkpoint
  resume_state: ~
```

Training outputs are saved to `experiments/train_KD_RepSR_x4/`.

---

## 5. Convert to Deployment Weights

After training, fuse all multi-branch RepSRBlocks into single 3x3 convolutions
using `scripts/convert_rep_sr.py`.

```bash
python scripts/convert_rep_sr.py \
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \
    --output experiments/converted/RepSR_x4_deployed.pth \
    --opt    options/train/train_KD_RepSR_x4.yml
```

The script will:
1. Build the RepSR model from the YAML arch params.
2. Load the BasicSR-format checkpoint (auto-detects `params_ema` / `params` keys).
3. Call `model.reparameterize()` to fuse all branches.
4. Run a quick shape sanity check.
5. Save a plain state-dict `.pth` file.

**Without a YAML file** (manual param specification):

```bash
python scripts/convert_rep_sr.py \
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \
    --output experiments/converted/RepSR_x4_deployed.pth \
    --num_feat 64 --num_blocks 8 --upscale 4 \
    --use_space_to_depth false
```

**Load the deployed weights:**

```python
import torch
from hat.archs.rep_sr_arch import RepSR

model = RepSR(num_feat=64, num_blocks=8, upscale=4)
model.load_state_dict(torch.load('experiments/converted/RepSR_x4_deployed.pth'))
model.eval()
```

---

## 6. Final Inference / Test

Edit `options/test/test_KD_RepSR_x4.yml` and set the student checkpoint path:

```yaml
path:
  pretrain_network_g: ./experiments/converted/RepSR_x4_deployed.pth
  param_key_g: ~   # plain state-dict (no wrapper key)
```

Then run:

```bash
python hat/test.py -opt options/test/test_KD_RepSR_x4.yml
```

Results (SR images + PSNR/SSIM metrics) are saved to `results/test_KD_RepSR_x4/`.

**Tiling for large images:**
Uncomment the `tile:` block in the test YAML:

```yaml
tile:
  tile_size: 256
  tile_pad: 32
```

---

## 7. Architecture Notes

### RepSR

| Training | Inference (after reparameterize) |
|----------|----------------------------------|
| 3x3 Conv + BN | Single fused 3x3 Conv |
| 1x1 Conv + BN | (merged into above) |
| Identity + BN | (merged into above) |

- All branches are summed **before** the PReLU activation.
- `reparameterize()` is mathematically lossless.
- The `conv_body` layer (after the main block stack) is the KD feature hook point.

### Knowledge Distillation Pipeline

```
LR Input
   |
   +--- Teacher (frozen, eval) ---------------------------------+
   |       conv_before_upsample -> [hook] feat_T                |
   |       -> pseudo-GT (HR output)                             |
   |                                                             |
   +--- Student (trainable) ------------------------------------+
           conv_body -> [hook] feat_S                            |
           -> student output                                     |
                                                                 |
   feat_S --[1x1 projector]--> feat_S_proj                      |
                                                                 |
   Loss = lambda_feat * MSE(feat_S_proj, feat_T)                |
        + lambda_out  * L1(student_out, pseudo-GT) <------------+
        + lambda_pix  * L1(student_out, GT_HR)   [optional]
```

### MambaIRv2

Requires `pip install mamba-ssm einops`.
The architecture combines window-based attention (Swin-style) with Mamba
state-space models (ASSM) for efficient long-range dependency modelling.
Set `type: MambaIRv2` in `network_teacher:` to use it as the teacher.

### File Map

| File | Role |
|------|------|
| `hat/archs/rep_sr_arch.py` | RepSR student architecture |
| `hat/archs/mambairv2_arch.py` | MambaIRv2 teacher architecture |
| `hat/models/kd_sr_model.py` | KD training + validation model |
| `options/train/train_KD_RepSR_x4.yml` | Training configuration |
| `options/test/test_KD_RepSR_x4.yml` | Test / inference configuration |
| `scripts/convert_rep_sr.py` | Reparameterization & export script |

---

## Citation

If you use this codebase, please cite the original works:

```bibtex
@inproceedings{chen2023hat,
  title={Activating More Pixels in Image Super-Resolution Transformer},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
  booktitle={CVPR},
  year={2023}
}

@article{guo2024mambairv2,
  title={MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author={Guo, Hang and Li, Jinmin and Dai, Tao and Ouyang, Zhihao and Ren, Xudong and Xia, Shu-Tao},
  year={2024}
}
```
