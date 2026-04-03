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
│   ├── my_lr_images/             # Real LR images (any resolution, no HR needed)
│   ├── Set5/
│   │   ├── GTmod4/               # HR ground-truth (validation only)
│   │   └── LRbicx4/              # LR bicubic inputs (validation)
│   └── Set14/
│       ├── GTmod4/
│       └── LRbicx4/
└── experiments/
    └── pretrained_models/
        └── HAT_SRx4_ImageNet-pretrain.pth   # teacher checkpoint
```

> **No HR images required for KD training.**
> `SingleLRDataset` loads LR images as-is and the frozen teacher generates
> Pseudo-GT HR patches on-the-fly during training.

### Obtaining real LR images

Any collection of natural LR images works. Example sources:

- Extract LR sub-images from an existing dataset (e.g. DF2K LR bicubic):

  ```bash
  python basicsr/scripts/extract_subimages.py \
      --input  datasets/DF2K/DF2K_LR_bicubic/X4 \
      --output datasets/my_lr_images \
      --n_thread 20
  ```

- Or simply point `dataroot_lq` at your own folder of LR `.png` / `.jpg` files.

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

### 3-e. SingleLRDataset (LR-only training)

Use `SingleLRDataset` when you have only LR images and want the teacher to
generate Pseudo-GT on-the-fly.  **No bicubic downsampling is applied — the
LR images are used at their native resolution.**

```yaml
datasets:
  train:
    name: LR_train
    type: SingleLRDataset          # <-- use this instead of ImageNetPairedDataset
    dataroot_lq: datasets/my_lr_images   # folder of real LR images
    io_backend:
      type: disk

    lq_patch_size: 64   # random-crop size in LR space (no resize)
    use_hflip: true
    use_rot: true

    # Remove / omit these keys (they are NOT needed):
    #   dataroot_gt: ...         (no HR images required)
    #   gt_size: ...             (no resize)
    #   scale: ...               (no downsampling)
```

**Key differences from `ImageNetPairedDataset`:**

| Feature | `ImageNetPairedDataset` | `SingleLRDataset` |
|---|---|---|
| Input | HR images (downsampled to LR on-the-fly) | Real LR images |
| Resize | Bicubic downscale by 1/scale | **None** |
| GT returned | Yes (HR image) | No (teacher generates it) |
| `dataroot_hr` | Required | Not used |

### 3-f. Tiling (`tile`)

Enable tiling for GPU-memory-limited **validation or inference**.
Tiling is **not** used during training (training always works on small patches).

```yaml
tile:
  patch_size: 256     # LR tile spatial size fed to the model
  overlap_size: 32    # LR-space overlap between adjacent tiles
                      # HR overlap = overlap_size * scale
```

**How the stitching works (linear blending):**

```
LR overlap:  overlap_size  pixels
HR overlap:  overlap_size * scale  pixels
Blend fade:  (overlap_size * scale) / 2  pixels at each tile border
```

Each HR output tile is weighted by a 2-D linear-ramp mask (weight = 1 in
the centre, linearly tapers to ~0 at the border).  Accumulated tiles are
normalised by the summed weight map, producing seamless blending without
visible seams even at tile boundaries.

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

## 7. End-to-End Experiment Example (x4 SR, HAT Teacher)

아래는 HAT teacher + RepSR student, scale×4, GPU 1장 기준으로 처음부터 끝까지 실행하는 전체 명령어 예시입니다.

### Step 1 — 학습 (KD Training)

```bash
# DF2K 데이터셋 준비 및 teacher 체크포인트 다운로드 후 실행
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml
```

학습 로그/체크포인트: `experiments/train_KD_RepSR_x4/`

학습 중 중간 검증 결과 (Set5 PSNR/SSIM)는 `5000 iter`마다 콘솔에 출력됩니다.

```
[train_KD_RepSR_x4 iter:5000] psnr: 28.42 dB  ssim: 0.8031
[train_KD_RepSR_x4 iter:10000] psnr: 28.91 dB  ssim: 0.8112
...
```

**Space-to-Depth 활성화 실험** (더 빠른 inference를 원할 때):

```yaml
# options/train/train_KD_RepSR_x4.yml 수정
network_g:
  type: RepSR
  num_feat: 64
  num_blocks: 8
  upscale: 4
  use_space_to_depth: true   # ← 변경
  s2d_factor: 2
```

```bash
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml
```

**MambaIRv2 Teacher로 전환** (mamba-ssm 설치 필요):

```yaml
# network_teacher 섹션 교체 (YAML 내 주석 해제)
network_teacher:
  type: MambaIRv2
  upscale: 4
  embed_dim: 60
  d_state: 8
  depths: [6, 6, 6, 6]
  num_heads: [4, 4, 4, 4]
  window_size: 16
  inner_rank: 32
  num_tokens: 64
  convffn_kernel_size: 5
  mlp_ratio: 2.
  img_range: 1.
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
```

```bash
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml
```

---

### Step 2 — 가중치 변환 (Reparameterization)

학습 완료 후 다중 브랜치를 단일 3×3 Conv로 융합합니다.

```bash
# 권장: YAML 기반 (아키텍처 파라미터 자동 인식)
python scripts/convert_rep_sr.py \
    --input  experiments/train_KD_RepSR_x4/models/net_g_400000.pth \
    --output experiments/converted/RepSR_x4_deployed.pth \
    --opt    options/train/train_KD_RepSR_x4.yml
```

예상 출력:
```
[convert_rep_sr] Architecture parameters:
  num_feat: 64, num_blocks: 8, upscale: 4, use_space_to_depth: False
[convert_rep_sr] Auto-detected param key: "params_ema"
[convert_rep_sr] Reparameterizing RepSR blocks ...
[convert_rep_sr] Done. All RepSRBlocks fused into single Conv2d.
[convert_rep_sr] Sanity check passed. Output shape: (1, 3, 256, 256)
[convert_rep_sr] Saved deployed checkpoint to: experiments/converted/RepSR_x4_deployed.pth
[convert_rep_sr] Model parameters: 0.412 M
```

---

### Step 3 — 최종 추론 (Inference / Test)

```bash
# test YAML의 체크포인트 경로를 변환된 파일로 수정 후 실행
python hat/test.py -opt options/test/test_KD_RepSR_x4.yml
```

결과 이미지 및 PSNR/SSIM: `results/test_KD_RepSR_x4/`

**메모리 제한 GPU에서 타일 추론:**

```bash
# options/test/test_KD_RepSR_x4.yml 에서 tile 블록 주석 해제:
#   tile:
#     tile_size: 256
#     tile_pad: 32
python hat/test.py -opt options/test/test_KD_RepSR_x4.yml
```

---

## 8. Architecture Notes

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
| `hat/data/single_lr_dataset.py` | LR-only dataset (no resize, teacher generates GT) |
| `hat/models/kd_sr_model.py` | KD training + tiled inference model |
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
