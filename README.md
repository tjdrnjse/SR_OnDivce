# HAT Knowledge Distillation SR Pipeline

HAT 기반의 Knowledge Distillation Super-Resolution 프레임워크. 대형 Teacher 모델(HAT)의 지식을 경량 Student 모델(RepSR / HAT-S)에 전달하여 빠른 추론 속도를 유지하면서 화질을 극대화한다.

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [데이터셋 구성](#3-데이터셋-구성)
4. [KD Loss 조합](#4-kd-loss-조합)
5. [학습 실행](#5-학습-실행)
6. [Joint-Batch 학습 (다중 스트림)](#6-joint-batch-학습-다중-스트림)
7. [Cross-Scale KD](#7-cross-scale-kd)
8. [Tiling 추론 (검증 및 Teacher 단독)](#8-tiling-추론-검증-및-teacher-단독)
9. [ONNX 변환 (NPU 포팅)](#9-onnx-변환-npu-포팅)
10. [RepSR 가중치 변환 (배포용)](#10-repsr-가중치-변환-배포용)
11. [최종 추론 / 테스트](#11-최종-추론--테스트)
12. [YAML 주요 키 레퍼런스](#12-yaml-주요-키-레퍼런스)
13. [파일 맵](#13-파일-맵)

---

## 1. 환경 설정

```bash
# 저장소 클론 및 의존성 설치
git clone <this-repo>
cd HAT
pip install -r requirements.txt
pip install basicsr>=1.4.2

# 편집 가능 패키지로 설치 (hat.* 임포트 경로 활성화)
python setup.py develop
```

---

## 2. 아키텍처 개요

### KD 파이프라인

```
LR 입력
  │
  ├── Teacher (HAT-Large, 동결) ──────────────────┐
  │     conv_before_upsample → [hook] feat_T      │
  │     → pseudo-GT (HR 출력)                      │
  │                                                │
  └── Student (RepSR / HAT-S, 학습 가능) ──────────┤
        conv_body / conv_before_upsample → feat_S  │
        → student 출력                              │
                                                   │
  feat_S ──[1×1 Projector]──▶ feat_S_proj          │
                                                   │
  Loss = w_feat × MSE(feat_S_proj, feat_T)         │
       + w_out  × L1(student_out, pseudo-GT) ◀────┘
       + w_pix  × L1(student_out, GT_HR)   [선택]
```

### Teacher: HAT (Hybrid Attention Transformer)

| 항목 | 값 |
|---|---|
| 아키텍처 | Window Attention + Channel Attention |
| Feature hook 위치 | `conv_before_upsample` (출력: B×64×H×W) |
| 역할 | 동결 상태로 pseudo-GT 생성 및 feature 제공 |

### Student: RepSR

| 항목 | 값 |
|---|---|
| 아키텍처 | 다중 브랜치 Conv (학습 시) → 단일 3×3 Conv (추론 시) |
| Feature hook 위치 | `conv_body` |
| Space-to-Depth | `use_space_to_depth: true` 시 PixelUnshuffle 전처리 |
| 가중치 변환 | `scripts/convert_rep_sr.py` 로 reparameterize 후 배포 |

### Student: HAT-S (HAT 기반 Student)

Feature hook 위치가 `conv_before_upsample`이며, Teacher(HAT-L)와 동일 구조의 소형 버전으로 사용 가능.

---

## 3. 데이터셋 구성

학습 데이터셋은 세 가지 타입을 선택 또는 혼합할 수 있다.

### 3-a. PairedImageDataset (HR+LR 쌍)

HR과 LR이 디스크에 쌍으로 존재하는 표준 SR 데이터셋.
Supervised pixel loss (`pixel_opt`)와 함께 사용 가능.

```yaml
datasets:
  train:
    name: DF2K
    type: PairedImageDataset
    dataroot_gt: datasets/DF2K/DF2K_HR_sub
    dataroot_lq: datasets/DF2K/DF2K_bicx4_sub
    io_backend:
      type: disk
    gt_size: 256       # HR 패치 크기 (LR = 256 / scale)
    use_hflip: true
    use_rot: true
    batch_size_per_gpu: 16
    num_worker_per_gpu: 6
```

### 3-b. SingleLRDataset (LR 전용)

HR 이미지 없이 실제 LR 이미지만 사용. Teacher가 pseudo-GT를 on-the-fly로 생성하므로 HR 불필요.

```yaml
datasets:
  train:
    name: LR_real
    type: SingleLRDataset
    dataroot_lq: datasets/my_lr_images
    io_backend:
      type: disk
    lq_patch_size: 64   # LR 공간에서의 랜덤 크롭 크기
    use_hflip: true
    use_rot: true
```

> `pixel_opt`는 GT가 없으므로 반드시 비활성화(주석 처리)해야 한다.

### 3-c. RealESRGANDataset (HR 전용 + GPU 열화 합성)

HR 이미지만 있으면 LQ를 GPU에서 2단계 열화 파이프라인(blur → noise → JPEG)으로 on-the-fly 합성. `degradation:` 섹션의 파라미터로 열화 강도를 조절한다.

```yaml
datasets:
  train_1:
    name: HR_degradation
    type: RealESRGANDataset
    dataroot_gt: datasets/hr_images
    io_backend:
      type: disk
    use_hflip: true
    use_rot: true
    blur_kernel_size: 21
    # ... (자세한 kernel 설정은 train_KD_HAT_S_x4_joint.yml 참조)
    batch_size_per_gpu: 16
    num_worker_per_gpu: 4

degradation:
  gt_size: 256           # HR 패치 → LQ = gt_size // teacher_upscale
  resize_prob: [0.2, 0.7, 0.1]
  resize_range: [0.15, 1.5]
  gaussian_noise_prob: 0.5
  noise_range: [1, 30]
  jpeg_range: [30, 95]
  # 2단계 열화 파라미터는 옵션 YAML 참조
```

---

## 4. KD Loss 조합

`kd_feat_opt`와 `kd_output_opt`의 `loss_weight`로 각 Loss 컴포넌트를 독립적으로 제어한다. `0.0`이거나 키 자체를 생략하면 해당 컴포넌트는 완전히 비활성화(오버헤드 없음)된다.

### 모드 A — Output KD만 (기본 경량 설정)

```yaml
train:
  kd_output_opt:
    loss_weight: 1.0
  # kd_feat_opt 생략 또는 loss_weight: 0.0
```

Student 출력과 Teacher pseudo-GT 간 L1 Loss만 사용. Feature projector가 빌드되지 않아 메모리/연산 절약.

### 모드 B — Feature KD만 (중간 레이어 감독)

```yaml
train:
  kd_feat_opt:
    loss_weight: 1.0
    student_feat_channels: 64   # network_g.num_feat 와 일치해야 함
    teacher_feat_channels: 64   # HAT conv_before_upsample = 64 고정
  # kd_output_opt 생략 또는 loss_weight: 0.0
```

Student 중간 feature를 1×1 Conv로 프로젝션하여 Teacher feature와 MSE.

### 모드 C — Feature + Output KD (권장, 최강 감독)

```yaml
train:
  kd_feat_opt:
    loss_weight: 1.0
  kd_output_opt:
    loss_weight: 1.0
  student_feat_channels: 64
  teacher_feat_channels: 64
```

### Supervised Pixel Loss 추가 (선택)

GT가 있는 경우(`PairedImageDataset` 또는 Stream A-bp) 추가 가능.

```yaml
train:
  pixel_opt:
    type: L1Loss
    loss_weight: 0.5
    reduction: mean
```

Joint-Batch에서 Stream별 적용 기준:

| 스트림 | pixel_opt 적용 대상 |
|---|---|
| A-deg (on-the-fly 합성) | USM-sharpened GT |
| A-bp (paired LQ bypass) | 실제 HR GT |
| B (LR only) | 미적용 (GT 없음) |

---

## 5. 학습 실행

### 기본 학습 (단일 GPU)

```bash
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml
```

### 다중 GPU (DDP)

```bash
# torchrun (PyTorch >= 2.0 권장)
torchrun --standalone --nproc_per_node=4 \
    hat/train.py -opt options/train/train_KD_RepSR_x4.yml --launcher pytorch

# 이전 방식 (torch.distributed.launch)
python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=4321 \
    hat/train.py -opt options/train/train_KD_RepSR_x4.yml --launcher pytorch
```

### 재개 (Resume)

```bash
python hat/train.py -opt options/train/train_KD_RepSR_x4.yml --auto_resume
```

### 제공 학습 YAML

| 파일 | 설명 |
|---|---|
| `train_KD_RepSR_x4.yml` | RepSR Student + HAT Teacher, x4, PairedImageDataset |
| `train_KD_RepSR_x3_10MB.yml` | RepSR Student (~10MB), x3, SingleLRDataset ×2 |
| `train_KD_HAT_S_x4_joint.yml` | HAT-S Student + HAT-L Teacher, Joint-Batch (2-stream) |
| `train_KD_three_stream_example.yml` | 3-stream (A-deg + A-bp + B) 예시 |
| `train_KD_HAT_S_x3_from_HAT_GAN_x4.yml` | HAT-S x3, GAN Teacher |

---

## 6. Joint-Batch 학습 (다중 스트림)

서로 다른 데이터셋을 하나의 미니배치에 혼합하여 학습. `JointBatchSampler`가 매 배치에서 A:B = 1:1 비율을 보장.

### 스트림 구성

| 스트림 | 데이터셋 타입 | LQ 생성 방식 | Loss |
|---|---|---|---|
| **A-deg** | RealESRGANDataset (HR only) | GPU on-the-fly 2단계 열화 | KD + pixel vs USM-GT |
| **A-bp** | RealESRGANDataset (HR+LR paired, 확률적) | 디스크에서 직접 로드 | KD + supervised pixel vs GT |
| **B** | SingleLRDataset | 없음 (실제 LR 그대로) | KD only |

- A-deg / A-bp 분리는 `RealESRGANDataset`의 `prob_paired_lq` 파라미터로 조절
- `prob_paired_lq: 0.0` → 전체 Stream A가 A-deg (열화 합성 전용)
- `prob_paired_lq: 0.3` → A 샘플의 30%가 A-bp (supervised), 70%가 A-deg

### 2-stream 설정 (A-deg + B)

```yaml
joint_batch_training: true

datasets:
  train_1:              # Stream A: RealESRGANDataset
    type: RealESRGANDataset
    dataroot_gt: /path/to/hr_images
    batch_size_per_gpu: 16   # 8 from A + 8 from B

  train_2:              # Stream B: SingleLRDataset
    type: SingleLRDataset
    dataroot_lq: /path/to/lr_images
    lq_patch_size: 64   # = degradation.gt_size // scale

degradation:
  gt_size: 256
  # ...
```

### 3-stream 설정 (A-deg + A-bp + B)

```yaml
joint_batch_training: true

datasets:
  train_1:
    type: RealESRGANDataset
    dataroot_gt: /path/to/hr_images
    dataroot_lq: /path/to/paired_lr_images   # A-bp용 LR 폴더 추가
    prob_paired_lq: 0.3                       # A-bp 발생 확률
    scale: 4
    # ...

  train_2:
    type: SingleLRDataset
    dataroot_lq: /path/to/lr_only_images
    lq_patch_size: 64
```

> 전체 설정은 `options/train/train_KD_three_stream_example.yml` 참조.

### DDP 학습 (Joint-Batch)

```bash
torchrun --standalone --nproc_per_node=8 \
    hat/train.py \
    -opt options/train/train_KD_HAT_S_x4_joint.yml \
    --launcher pytorch
```

### TensorBoard 모니터링

```bash
tensorboard --logdir experiments/train_KD_HAT_S_x4_joint/tb_logger --port 6006
```

| TensorBoard 항목 | 설명 |
|---|---|
| `train/lq_bicubic` | 학습 LQ (bilinear 업샘플) |
| `train/teacher_pseudo_gt` | Teacher pseudo-GT |
| `train/student_output` | Student 출력 |
| `l_kd_feat` / `l_kd_out` / `l_pix` / `l_total` | 손실 곡선 |

---

## 7. Cross-Scale KD

Teacher와 Student의 upscale 배수가 다른 경우 자동으로 pseudo-GT를 bicubic으로 리사이즈하여 맞춘다.

```yaml
network_teacher:
  type: HAT
  upscale: 4        # Teacher: ×4

network_g:
  type: RepSR
  upscale: 3        # Student: ×3 (Teacher > Student 이어야 함)
```

- Teacher upscale < Student upscale인 경우 오류 발생 (방향이 반대이므로 불가)
- 예: `train_KD_HAT_S_x3_from_HAT_GAN_x4.yml`에서 Teacher(x4) → Student(x3)

---

## 8. Tiling 추론 (검증 및 Teacher 단독)

### 검증 시 Tiling

메모리 제한 GPU에서 큰 이미지를 검증할 때 사용. 학습 중에는 작동하지 않으며 검증/테스트 전용.

```yaml
tile:
  patch_size: 256    # LR 타일 크기
  overlap_size: 32   # LR 타일 간 겹침 (HR 겹침 = overlap_size × scale)
```

블렌딩 방식: 선형 램프 가중치 마스크로 HR 타일을 누적 후 정규화. 경계 아티팩트 없음.

### Teacher 단독 추론

KD 파이프라인 없이 Teacher만 독립적으로 SR 추론.

```bash
python scripts/inference_teacher_tiling.py \
    --opt    options/inference/teacher_hat_x3.yml \
    --input  datasets/my_lr_images \
    --output results/teacher_hat_x3
```

CLI 옵션:

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--opt` | 필수 | inference YAML 경로 |
| `--input` | YAML `input_dir` | LR 이미지 폴더 |
| `--output` | YAML `output_dir` | SR 결과 저장 폴더 |
| `--device` | auto | `cuda` / `cpu` |
| `--suffix` | `_SR` | 출력 파일명 접미사 |

YAML 예시 (`options/inference/teacher_hat_x3.yml`):

```yaml
scale: 3
tile_size: 256
tile_overlap: 32
input_dir:  datasets/my_lr_images
output_dir: results/teacher_hat_x3
device: cuda

network_t:
  type: HAT
  upscale: 3
  img_size: 64
  window_size: 16
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  upsampler: 'pixelshuffle'
  # ...

path:
  pretrain_network_t: experiments/pretrained_models/HAT_SRx3_ImageNet-pretrain.pth
```

---

## 9. ONNX 변환 (NPU 포팅)

Teacher 모델을 고정 공간 크기로 ONNX로 내보낸다. NPU 컴파일러에 제출하여 연산자 호환성 검증 용도.

```bash
pip install onnx onnxruntime
```

```bash
# 기본 변환 (입력 크기 64×64)
python scripts/export_onnx.py \
    --opt options/inference/teacher_hat_x3.yml

# 입력 크기 지정 (window_size=16의 배수여야 함)
python scripts/export_onnx.py \
    --opt        options/inference/teacher_hat_x3.yml \
    --output-dir results/onnx \
    --filename   hat_sr_x3.onnx \
    --opset      17 \
    --input-h    256 \
    --input-w    256 \
    --device     cpu
```

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--opt` | 필수 | inference YAML (`network_t` 섹션 사용) |
| `--output-dir` | `results/onnx` | `.onnx` 저장 폴더 |
| `--filename` | `<type>_x<scale>.onnx` | 출력 파일명 |
| `--opset` | `17` | ONNX opset 버전 |
| `--input-h` / `--input-w` | `64` | LR 입력 크기 (window_size 배수) |
| `--device` | `cpu` | tracing 장치 |

> - 입력 크기는 내보내기 시 고정됨 (NPU 컴파일러 요구사항)
> - 체크포인트가 없어도 내보내기 가능 (그래프 구조 확인 용도)
> - `onnx` 설치 시 내보내기 후 자동으로 `check_model` 실행

### ONNX 모델 테스트

```bash
python scripts/test_onnx.py \
    --onnx   results/onnx/HAT_x3.onnx \
    --input  datasets/my_lr_images \
    --output results/onnx_test
```

입력 이미지 크기가 ONNX 모델의 기대 크기와 다르면 Lanczos 리사이즈 후 추론하며, 리사이즈된 입력도 함께 저장된다.

---

## 10. RepSR 가중치 변환 (배포용)

학습된 RepSR 체크포인트의 다중 브랜치를 단일 3×3 Conv로 융합. 수학적으로 동일하며 추론 속도 향상.

```bash
# YAML 기반 (아키텍처 파라미터 자동 인식, 권장)
python scripts/convert_rep_sr.py \
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \
    --output experiments/converted/RepSR_x4_deployed.pth \
    --opt    options/train/train_KD_RepSR_x4.yml

# 수동 파라미터 지정
python scripts/convert_rep_sr.py \
    --input  experiments/train_KD_RepSR_x4/models/net_g_latest.pth \
    --output experiments/converted/RepSR_x4_deployed.pth \
    --num_feat 64 --num_blocks 8 --upscale 4 \
    --use_space_to_depth false
```

변환된 가중치 로드:

```python
import torch
from hat.archs.rep_sr_arch import RepSR

model = RepSR(num_feat=64, num_blocks=8, upscale=4)
model.load_state_dict(torch.load('RepSR_x4_deployed.pth'))
model.eval()
```

---

## 11. 최종 추론 / 테스트

```yaml
# options/test/test_KD_RepSR_x4.yml
path:
  pretrain_network_g: ./experiments/converted/RepSR_x4_deployed.pth
  param_key_g: ~    # plain state-dict이면 ~ 로 설정
```

```bash
python hat/test.py -opt options/test/test_KD_RepSR_x4.yml
```

결과 (SR 이미지 + PSNR/SSIM): `results/test_KD_RepSR_x4/`

---

## 12. YAML 주요 키 레퍼런스

### 최상위

| 키 | 설명 | 예시 |
|---|---|---|
| `model_type` | 모델 클래스 | `KDSRModel` |
| `scale` | SR 배율 | `4` |
| `num_gpu` | GPU 수 | `auto` |
| `joint_batch_training` | Joint-Batch 활성화 | `true` |

### `network_g` (Student)

| 키 | 설명 |
|---|---|
| `type` | `RepSR` 또는 `HAT` |
| `num_feat` | 채널 수 (RepSR) |
| `num_blocks` | 블록 수 (RepSR) |
| `upscale` | SR 배율 |
| `use_space_to_depth` | S2D 전처리 활성화 |
| `s2d_factor` | PixelUnshuffle 배율 (S2D 사용 시) |

### `network_teacher` (Teacher)

| 키 | 설명 |
|---|---|
| `type` | `HAT` |
| `upscale` | SR 배율 (≥ student upscale) |
| `embed_dim` | 임베딩 차원 (HAT-base=180, HAT-S=144) |
| `depths` | 각 스테이지 레이어 수 |
| `window_size` | Attention 윈도우 크기 |

### `train`

| 키 | 설명 |
|---|---|
| `kd_feat_opt.loss_weight` | Feature KD 가중치 (0=비활성) |
| `kd_output_opt.loss_weight` | Output KD 가중치 (0=비활성) |
| `pixel_opt.loss_weight` | Supervised L1 가중치 (0=비활성) |
| `student_feat_channels` | Student feature 채널 수 (`num_feat` 와 일치) |
| `teacher_feat_channels` | Teacher feature 채널 수 (HAT 고정 64) |

### `tile` (검증/추론용 Tiling)

| 키 | 설명 |
|---|---|
| `patch_size` | LR 타일 크기 (window_size 배수) |
| `overlap_size` | LR 타일 간 겹침 크기 |

### `degradation` (RealESRGAN 스트림 전용)

| 키 | 설명 |
|---|---|
| `gt_size` | HR 패치 크기 (LQ = gt_size ÷ teacher_upscale) |
| `resize_prob` | 리사이즈 방향 확률 [up, down, keep] |
| `gaussian_noise_prob` | 가우시안 노이즈 적용 확률 |
| `jpeg_range` | JPEG 압축 품질 범위 |

---

## 13. 파일 맵

| 파일 | 역할 |
|---|---|
| `hat/archs/hat_arch.py` | HAT 아키텍처 |
| `hat/archs/rep_sr_arch.py` | RepSR Student 아키텍처 |
| `hat/models/kd_sr_model.py` | KD 학습 루프, Tiling 추론, Joint-Batch 열화 |
| `hat/data/single_lr_dataset.py` | LR 전용 데이터셋 |
| `hat/data/realesrgan_dataset.py` | RealESRGAN 스타일 데이터셋 (HR + blur kernel) |
| `hat/data/joint_batch_sampler.py` | `JointBatchSampler`, `StreamTaggedDataset`, `joint_collate_fn` |
| `options/train/train_KD_RepSR_x4.yml` | RepSR x4 KD 학습 설정 |
| `options/train/train_KD_RepSR_x3_10MB.yml` | RepSR x3 (~10MB) KD 학습 설정 |
| `options/train/train_KD_HAT_S_x4_joint.yml` | HAT-S x4 Joint-Batch 학습 설정 |
| `options/train/train_KD_three_stream_example.yml` | 3-stream 학습 예시 설정 |
| `options/inference/teacher_hat_x3.yml` | HAT x3 Teacher 단독 추론 설정 |
| `scripts/convert_rep_sr.py` | RepSR 가중치 Reparameterization |
| `scripts/inference_teacher_tiling.py` | Teacher 단독 Tiling SR 추론 |
| `scripts/export_onnx.py` | Teacher → ONNX 변환 |
| `scripts/test_onnx.py` | ONNX 모델 SR 추론 테스트 |

---

## Citation

```bibtex
@inproceedings{chen2023hat,
  title={Activating More Pixels in Image Super-Resolution Transformer},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
  booktitle={CVPR},
  year={2023}
}
```
