"""
최소 CPU Forward-Pass 검증 스크립트
====================================
PC 사양이 낮은 CPU 환경에서 모델 구조가 정상적으로 동작하는지만 확인합니다.
- 배치 크기: 1
- 공간 해상도: 16×16
- CUDA 불필요 (강제 CPU 모드)

사용법:
    python test_cpu_minimal.py

    # hat/ 아키텍처 파일이 삭제된 경우 먼저 복구:
    #   git restore hat/
    #   python test_cpu_minimal.py
"""

# ── 단계 1: basicsr 호환성 패치 (반드시 가장 먼저 실행) ──────────────────────
import basicsr_compat
basicsr_compat.apply()
# ─────────────────────────────────────────────────────────────────────────────

import sys
import traceback
import torch
import torch.nn as nn


# ─── CPU 강제 설정 ─────────────────────────────────────────────────────────
torch.set_num_threads(1)        # CPU 코어 1개로 제한 → 메모리 안전
torch.set_grad_enabled(False)   # 추론 전용: autograd 그래프 비활성화


# ─── 유틸 ──────────────────────────────────────────────────────────────────
SEP = "=" * 55


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def shape_str(t: torch.Tensor) -> str:
    return f"[{', '.join(str(d) for d in t.shape)}]  dtype={t.dtype}"


def check_shape(tag: str, actual: torch.Tensor, expected_shape: tuple):
    ok = tuple(actual.shape) == expected_shape
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {tag}")
    print(f"         기대값: {list(expected_shape)}")
    print(f"         실제값: {list(actual.shape)}")
    return ok


# ─── 모델별 검증 함수 ────────────────────────────────────────────────────────

def validate_rep_sr():
    """RepSR: 경량 학생 모델 CPU forward 검증."""
    section("RepSR (Student) -- CPU Forward Validation")

    try:
        from hat.archs.rep_sr_arch import RepSR
    except ImportError as e:
        print(f"  [SKIP] hat.archs.rep_sr_arch 를 import할 수 없습니다: {e}")
        print("         'git restore hat/' 을 실행한 뒤 다시 시도하세요.")
        return False

    # CPU 환경을 위해 채널/블록 수를 줄임
    model = RepSR(
        num_in_ch=3,
        num_feat=32,        # 원본 64 → 메모리 절약을 위해 32
        num_blocks=4,       # 원본 8  → 메모리 절약을 위해 4
        upscale=4,
        use_space_to_depth=False,
    )
    model.eval()

    B, C, H, W = 1, 3, 16, 16
    upscale = 4
    x = torch.randn(B, C, H, W)

    print(f"\n  입력 텐서:  {shape_str(x)}")

    try:
        y = model(x)
    except Exception:
        print("  [FAIL] forward() 실행 중 오류:")
        traceback.print_exc()
        return False

    print(f"  출력 텐서:  {shape_str(y)}")

    expected = (B, C, H * upscale, W * upscale)
    passed = check_shape("입출력 Shape", y, expected)

    if passed:
        print(f"\n  결론: RepSR forward pass 정상 완료.")
        print(f"        {H}x{W} -> {H*upscale}x{W*upscale}  (x{upscale} SR)")

    return passed


def validate_hat():
    """HAT (Teacher): window_size=4 축소 구성으로 16x16 CPU forward 검증."""
    section("HAT (Teacher) -- CPU Forward Validation (lightweight config)")

    try:
        from hat.archs.hat_arch import HAT
    except ImportError as e:
        print(f"  [SKIP] hat.archs.hat_arch 를 import할 수 없습니다: {e}")
        print("         'git restore hat/' 을 실행한 뒤 다시 시도하세요.")
        return False

    # 16×16 입력에서 window_size=4 → 4로 나누어 떨어짐, 최소 구성
    model = HAT(
        img_size=16,
        patch_size=1,
        in_chans=3,
        embed_dim=48,           # 원본 180 → 최소
        depths=(2, 2),          # 원본 (6,6,6,6,6,6) → 최소
        num_heads=(2, 2),
        window_size=4,          # 원본 16 → 16×16 입력에 맞게 축소
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2.,
        upscale=4,
        img_range=1.,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    model.eval()

    B, C, H, W = 1, 3, 16, 16
    upscale = 4
    x = torch.randn(B, C, H, W)

    print(f"\n  입력 텐서:  {shape_str(x)}")

    try:
        y = model(x)
    except Exception:
        print("  [FAIL] forward() 실행 중 오류:")
        traceback.print_exc()
        return False

    print(f"  출력 텐서:  {shape_str(y)}")

    expected = (B, C, H * upscale, W * upscale)
    passed = check_shape("입출력 Shape", y, expected)

    if passed:
        print(f"\n  결론: HAT forward pass 정상 완료.")
        print(f"        {H}x{W} -> {H*upscale}x{W*upscale}  (x{upscale} SR)")

    return passed


# ─── 메인 ──────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  CPU 최소 Forward-Pass 검증")
    print(SEP)
    print(f"  PyTorch  : {torch.__version__}")
    try:
        import torchvision
        print(f"  torchvision: {torchvision.__version__}")
    except ImportError:
        print("  torchvision: 미설치")
    print(f"  CPU 스레드: {torch.get_num_threads()}")

    results = {}
    results['RepSR'] = validate_rep_sr()
    results['HAT']   = validate_hat()

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    section("검증 결과 요약")
    all_pass = True
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {status:4s}  {name}")

    print()
    if all_pass:
        print("  모든 모델이 CPU에서 정상적으로 Forward Pass를 통과했습니다.")
    else:
        print("  일부 모델에서 문제가 발생했습니다. 위 출력을 확인하세요.")
        sys.exit(1)


if __name__ == '__main__':
    main()
