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
5. [Joint-Batch KD Training (H100 DDP)](#5-joint-batch-kd-training-h100-ddp)
6. [Convert to Deployment Weights](#6-convert-to-deployment-weights)
7. [Export to ONNX (NPU Porting)](#7-export-to-onnx-npu-porting)
8. [Test ONNX Model (NPU Verification)](#8-test-onnx-model-npu-verification)
9. [Final Inference / Test](#9-final-inference--test)
10. [Teacher Model Standalone Testing (Tiling Inference)](#10-teacher-model-standalone-testing-tiling-inference)
11. [Architecture Notes](#12-architecture-notes)

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

### 4-a. x4 모델 (기존)

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

### 4-b. x3 Mobile 10MB 모델 (`train_KD_RepSR_x3_10MB.yml`)

x3 스케일, **~10 MB (FP32) / ~2.5 M 파라미터** 크기의 Mobile-target RepSR 학습 설정입니다.

```bash
python hat/train.py -opt options/train/train_KD_RepSR_x3_10MB.yml
```

**실행 전 YAML에서 경로 변경 필요:**

```yaml
path:
  pretrain_network_g: /path/to/student_pretrained.pth   # 또는 ~ (scratch)
  pretrain_network_t: /path/to/hat_x3_imagenet_pretrained.pth  # 필수

datasets:
  train_1:
    dataroot_lq: /your/real/lr_dataset_1
  train_2:
    dataroot_lq: /your/real/lr_dataset_2
```

**주요 설계 의도:**

| 옵션 | 값 | 이유 |
|---|---|---|
| `scale` | 3 | x3 SR 태스크 (모바일 디스플레이 업스케일 대상) |
| `num_feat` / `num_blocks` | 128 / 10 | FP32 ~10 MB, on-device 추론 가능한 최대 크기 |
| `use_space_to_depth` | false | 학습 단순화; S2D는 추후 실험에서 별도 비교 |
| `lq_patch_size` | 256 | 큰 수용 영역(receptive field) 확보로 텍스처 복원력 향상 |
| `SingleLRDataset` × 2 | train_1 + train_2 | HR 이미지 없이 두 도메인의 실제 LR 이미지 동시 활용; Teacher가 Pseudo-GT 생성 |
| `strict_load_g` | false | 이전 체크포인트와 아키텍처 구조가 다소 달라도 부분 로드 허용 |
| `pixel_opt` | 주석 처리 | `SingleLRDataset`은 GT를 제공하지 않으므로 KD loss만 사용 |

Training outputs: `experiments/train_KD_RepSR_x3_10MB/`

---

---

## 5. Joint-Batch KD Training (H100 DDP)

두 가지 성격이 다른 데이터 스트림(Stream A: 열화 합성 + Stream B: 실제 LR)을 동시에 활용하는 **Joint-Batch KD 학습** 방법입니다.

### 개요

| | Stream A | Stream B |
|---|---|---|
| 데이터셋 타입 | `RealESRGANDataset` | `SingleLRDataset` |
| 입력 | HR 이미지 | 생성형 모델의 실제 LR 이미지 |
| Degradation | GPU에서 on-the-fly 합성 (blur → noise → JPEG) | **없음** (원본 그대로 사용) |
| Loss | KD Feature + KD Output + (옵션) Pixel L1 vs GT | KD Feature + KD Output |
| 목적 | 일반 화질 복원 및 Teacher 모방 | VAE artifact 제거용 KD |

각 미니배치는 `JointBatchSampler`에 의해 **A : B = 1 : 1 비율**이 보장됩니다.

### 전제 조건

```bash
# basicsr >= 1.4.2 (DiffJPEG, USMSharp 포함 버전)
pip install basicsr --upgrade

# YAML 내 lq_patch_size == degradation.gt_size // teacher_upscale 일치 필수
# 예: gt_size=256, teacher x4 → lq_patch_size=64
```

### 데이터셋 구조

```
HAT/datasets/
├── hr_images/          # Stream A: HR 이미지 폴더 (RealESRGANDataset)
│   ├── img001.png
│   └── ...
├── generative_lr/      # Stream B: 생성형 모델 LR 이미지 폴더 (SingleLRDataset)
│   ├── gen001.png
│   └── ...
└── Set5/, Set14/, ...  # 검증 데이터셋
```

### YAML 준비 (`options/train/train_KD_HAT_S_x4_joint.yml`)

필수 경로 수정:

```yaml
datasets:
  train_1:
    dataroot_gt: /your/hr_images       # Stream A HR 이미지 폴더
  train_2:
    dataroot_lq: /your/generative_lr   # Stream B 실제 LR 이미지 폴더

path:
  pretrain_network_teacher: /your/Real_HAT_GAN_SRx4.pth  # 필수
  pretrain_network_g: ~   # HAT-S warmup 체크포인트 (없으면 scratch)
```

핵심 파라미터 (`degradation` 섹션은 **Stream A에만** 적용됩니다):

```yaml
joint_batch_training: true   # JointBatchSampler 활성화 (필수)

degradation:
  gt_size: 256               # HR 패치 크기 → LQ = 256 // 4 = 64
  # ... (이하 degradation 파라미터는 Stream B에 영향 없음)
```

### H100 단일 노드 8-GPU DDP 학습

```bash
# torchrun (권장, PyTorch ≥ 2.0)
torchrun \
    --standalone \
    --nproc_per_node=8 \
    hat/train.py \
    -opt options/train/train_KD_HAT_S_x4_joint.yml \
    --launcher pytorch
```

또는 `torch.distributed.launch` (구버전 호환):

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=4321 \
    hat/train.py \
    -opt options/train/train_KD_HAT_S_x4_joint.yml \
    --launcher pytorch
```

### H100 멀티 노드 학습 (slurm 환경)

```bash
#!/bin/bash
#SBATCH --job-name=kd_hat_joint
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    hat/train.py \
    -opt options/train/train_KD_HAT_S_x4_joint.yml \
    --launcher pytorch
```

### 재개 (Resume)

```bash
torchrun --standalone --nproc_per_node=8 \
    hat/train.py \
    -opt options/train/train_KD_HAT_S_x4_joint.yml \
    --launcher pytorch \
    --auto_resume
```

### 학습 중 모니터링

```bash
# TensorBoard (실험 폴더 안에 tb_logger/ 생성됨)
tensorboard --logdir experiments/train_KD_HAT_S_x4_joint/tb_logger --port 6006

# 학습 로그 실시간 확인
tail -f experiments/train_KD_HAT_S_x4_joint/train_train_KD_HAT_S_x4_joint_*.log
```

TensorBoard에서 확인 가능한 항목:
- `train/lq_bicubic`, `train/teacher_pseudo_gt`, `train/student_output` — 학습 샘플 시각화 (매 `tb_train_vis_freq` iter)
- `val/{dataset}/{img}_lq_bicubic`, `val/{dataset}/{img}_teacher`, `val/{dataset}/{img}_student` — 검증 시각화
- 손실: `l_kd_feat`, `l_kd_out`, `l_pix`, `l_total`

### 배치 구성 예시 (batch_size_per_gpu=16, world_size=8)

```
총 배치 크기 = 16 × 8 = 128 samples
  → Stream A (RealESRGAN) : 64 samples — HR + kernels → GPU degradation → LQ
  → Stream B (SingleLR)   : 64 samples — 실제 LR 이미지 그대로 사용

모델 forward:
  self.lq = cat(lq_A, lq_B)   # shape: (128, 3, 64, 64)
  pseudo_gt = teacher(lq)      # shape: (128, 3, 256, 256) → (128, 3, 256, 256)
  student_out = student(lq)    # shape: (128, 3, 256, 256)

손실:
  l_kd_feat = MSE(proj(feat_S), feat_T)         # 전체 128개 샘플
  l_kd_out  = L1(student_out, pseudo_gt)         # 전체 128개 샘플
  l_pix     = L1(student_out[:64], gt_usm[:64])  # Stream A 64개 샘플만
```

---

## 6. Convert to Deployment Weights

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

## 7. Export to ONNX (NPU Porting)

Use `scripts/export_onnx.py` to convert a teacher model to ONNX format.
This is intended for **mobile/NPU porting feasibility checks** — the exported
graph can be submitted to an NPU compiler (e.g. Samsung Exynos, Qualcomm AI)
to verify operator support before full porting work begins.

> **Note:** Only the `network_t` (teacher) section of the inference YAML is
> used. `network_g` (student) keys are ignored.
> MambaIRv2 cannot be exported to ONNX because it relies on CUDA-only
> `mamba_ssm` kernels. Use the HAT teacher YAML.

### Prerequisites

```bash
pip install onnx onnxruntime
```

### Basic usage

```bash
# Export HAT x3 — input size taken from tile_size in YAML (256x256 by default)
python scripts/export_onnx.py \
    --opt options/inference/teacher_hat_x3.yml
```

Output: `results/onnx/HAT_x3.onnx`

### Full option reference

```bash
python scripts/export_onnx.py \
    --opt        options/inference/teacher_hat_x3.yml \
    --output-dir results/onnx \
    --filename   hat_sr_x3.onnx \
    --opset      17 \
    --input-h    256 \
    --input-w    256 \
    --device     cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--opt` | (required) | Path to inference YAML (`options/inference/*.yml`) |
| `--output-dir` | `results/onnx` | Directory for the `.onnx` file |
| `--filename` | `<ModelType>_x<scale>.onnx` | Output filename |
| `--opset` | `17` | ONNX opset version |
| `--input-h` | `64` | LR input height (fixed at export, must be multiple of 16) |
| `--input-w` | `64` | LR input width (fixed at export, must be multiple of 16) |
| `--device` | `cpu` | Device used for tracing (`cpu` recommended) |

### Notes

- **Input shape is fixed** at export time (NPU compilers require static shapes).
  Default is 64×64. Use `--input-h` / `--input-w` to set the tile size the
  NPU will actually receive (e.g. 128, 256). Both dimensions must be multiples
  of `window_size` (16).
- **Larger-than-training inputs work.** HAT uses local window attention
  (window_size=16), so attention is computed inside fixed 16×16 windows
  regardless of total image size. Exporting with 128×128 or 256×256 is valid.
- A **pretrained checkpoint is not required**. If `path.pretrain_network_t`
  is missing or the file does not exist, the script exports with random weights
  (suitable for graph / operator compatibility checks).
- **`use_checkpoint` (gradient checkpointing) is automatically disabled**
  before export — this flag is incompatible with `torch.onnx.export`.
- If `onnx` is installed, `onnx.checker.check_model` is run automatically
  on the saved file. If `onnxruntime` is installed, you can verify inference:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession('results/onnx/HAT_x3.onnx',
                            providers=['CPUExecutionProvider'])
inp  = np.random.randn(1, 3, 256, 256).astype(np.float32)
out  = sess.run(None, {'input': inp})
print(out[0].shape)  # (1, 3, 768, 768) for x3
```

---

## 8. Test ONNX Model (NPU Verification)

Use `scripts/test_onnx.py` to run SR inference with a converted ONNX file.
This script is intended for NPU engineers to verify output quality before
compiling the model for the target device.

### Prerequisites

```bash
pip install onnxruntime Pillow
# For GPU inference:
pip install onnxruntime-gpu Pillow
```

### Usage

```bash
# Single image
python scripts/test_onnx.py \
    --onnx   results/onnx/HAT_x3.onnx \
    --input  path/to/lr_image.png \
    --output results/onnx_test

# Folder of images
python scripts/test_onnx.py \
    --onnx   results/onnx/HAT_x3.onnx \
    --input  datasets/my_lr_images \
    --output results/onnx_test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--onnx` | (required) | Path to the `.onnx` model file |
| `--input` | (required) | Input image file or folder of images |
| `--output` | `results/onnx_test` | Directory to save output images |

### Output files

| File | Description |
|------|-------------|
| `<stem>_sr.png` | Super-resolved output image |
| `<stem>_input_resized.png` | Resized input (only saved when resize was applied) |

### Resize behaviour

The ONNX model has a fixed expected input size (set at export time, e.g. 64×64).

- If the input image matches that size → inference runs directly.
- If the input image is a **different size** → it is resized (Lanczos) to the
  model's expected size before inference, and the resized version is saved
  alongside the SR output as `<stem>_input_resized.png`.

### Provider selection

GPU (CUDA) is used automatically when `onnxruntime-gpu` is installed and a
CUDA device is available. Falls back to CPU otherwise.

---

## 9. Final Inference / Test

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

## 10. Teacher Model Standalone Testing (Tiling Inference)

Use `scripts/inference_teacher_tiling.py` to evaluate a Teacher model
(HAT, MambaIRv2, etc.) **independently** -- without loading the student or
any KD components.

### How the tiling works (crop-and-paste)

```
LR image (arbitrary size)
  |
  v
Reflection-pad to fit tile grid  (stride = tile_size - tile_overlap)
  |
  +-- tile (0,0) --> Teacher --> HR tile --> crop c px per side --> paste
  +-- tile (0,1) --> Teacher --> HR tile --> crop c px per side --> paste
  ...
  v
Crop away the padding region  -->  Final HR image (W*scale x H*scale)

  c = (tile_overlap * scale) // 2   [HR border crop per side]
```

Cropping the border of each HR tile and pasting only the safe central region
eliminates boundary artifacts.  For edge tiles (first / last row / column),
the outer boundary is NOT cropped to preserve full image content.

---

### 7-a. HAT SR x3 (ImageNet Pretrained)

**YAML** (`options/inference/teacher_hat_x3.yml`):

```yaml
scale: 3

tile_size: 256      # LR tile size
tile_overlap: 32    # LR overlap  -->  HR crop = (32 * 3) // 2 = 48 px per side

input_dir:  datasets/my_lr_images
output_dir: results/teacher_hat_x3
device:     cuda

network_t:
  type: HAT
  upscale: 3
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'

path:
  pretrain_network_t: experiments/pretrained_models/HAT_SRx3_ImageNet-pretrain.pth
```

**Run:**

```bash
# Using the ready-made YAML
python scripts/inference_teacher_tiling.py \
    --opt   options/inference/teacher_hat_x3.yml \
    --input datasets/my_lr_images \
    --output results/teacher_hat_x3

# Override device or suffix on the fly
python scripts/inference_teacher_tiling.py \
    --opt    options/inference/teacher_hat_x3.yml \
    --input  datasets/Set5/LRbicx3 \
    --output results/teacher_hat_x3/Set5 \
    --device cuda:0 \
    --suffix _hat_x3
```

Output: `results/teacher_hat_x3/<image_name>_SR.png`

---

### 7-b. MambaIRv2 SR Small x3

> Reference: https://github.com/csguoh/MambaIR
>
> Requires: `pip install mamba-ssm einops` (CUDA GPU + matching toolkit)

**YAML** (`options/inference/teacher_mambairv2_small_x3.yml`):

```yaml
scale: 3

tile_size: 256
tile_overlap: 32    # HR crop = (32 * 3) // 2 = 48 px per side

input_dir:  datasets/my_lr_images
output_dir: results/teacher_mambairv2_small_x3
device:     cuda

# MambaIRv2 Small: embed_dim=60, 4-stage, depths=[6,6,6,6]
network_t:
  type: MambaIRv2
  upscale: 3
  img_size: 64
  in_chans: 3
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

path:
  pretrain_network_t: experiments/pretrained_models/MambaIRv2_Small_SRx3.pth
```

**Run:**

```bash
python scripts/inference_teacher_tiling.py \
    --opt   options/inference/teacher_mambairv2_small_x3.yml \
    --input datasets/my_lr_images \
    --output results/teacher_mambairv2_small_x3
```

---

### 7-c. CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--opt`    | (required) | Path to inference YAML |
| `--input`  | YAML `input_dir` | LR image folder |
| `--output` | YAML `output_dir` | Output folder for SR images |
| `--device` | YAML `device` / auto | `cuda`, `cuda:1`, `cpu` |
| `--suffix` | `_SR` | Suffix appended to each output filename stem |

---

### 7-d. YAML key reference

| Key | Description | Default |
|-----|-------------|---------|
| `scale` | SR upscale factor | `4` |
| `tile_size` | LR tile spatial size | `256` |
| `tile_overlap` | LR overlap between adjacent tiles | `32` |
| `input_dir` | Folder of LR input images | — |
| `output_dir` | Folder where SR PNGs are saved | `results/teacher_inference` |
| `device` | Compute device | auto-detect |
| `network_t` | Teacher architecture dict (`type` + params) | — |
| `path.pretrain_network_t` | Teacher checkpoint path | — |

---

## 11. End-to-End Experiment Example (x4 SR, HAT Teacher)

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

## 12. Architecture Notes

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
| `hat/data/joint_batch_sampler.py` | `JointBatchSampler`, `StreamTaggedDataset`, `joint_collate_fn` |
| `hat/models/kd_sr_model.py` | KD training + tiled inference + Joint-Batch degradation |
| `options/train/train_KD_RepSR_x4.yml` | Training config: RepSR student + HAT/MambaIRv2 teacher |
| `options/train/train_KD_HAT_S_x4_joint.yml` | Joint-Batch training config: HAT-S student + HAT-L teacher |
| `options/test/test_KD_RepSR_x4.yml` | Test / inference configuration |
| `scripts/convert_rep_sr.py` | Reparameterization & export script |
| `scripts/inference_teacher_tiling.py` | Teacher standalone SR with linear-blend tiling |
| `scripts/export_onnx.py` | Export teacher model to ONNX (fixed spatial size, NPU porting) |
| `scripts/test_onnx.py` | Run SR inference with a converted ONNX model (NPU verification) |
| `options/inference/teacher_hat_x3.yml` | Example YAML for HAT x3 teacher inference |
| `options/inference/teacher_mambairv2_small_x3.yml` | Example YAML for MambaIRv2 Small x3 teacher inference |

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
