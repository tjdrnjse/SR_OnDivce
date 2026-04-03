"""
BasicSR compatibility patch for torch==2.0.1 / torchvision==0.15.2

torchvision.transforms.functional_tensor 모듈 누락 에러를 런타임에 해결합니다.
반드시 basicsr (또는 basicsr에 의존하는 모든 모듈)을 import하기 *전에* 호출하세요.

Usage:
    import basicsr_compat
    basicsr_compat.apply()

    # 이후에 basicsr import 가능
    from basicsr.utils.registry import ARCH_REGISTRY
"""
import sys
import types


def apply():
    """등록된 모든 패치를 적용합니다."""
    _patch_functional_tensor()


def _patch_functional_tensor():
    """누락된 torchvision.transforms.functional_tensor 모듈을 주입합니다.

    일부 basicsr 빌드는 torchvision 버전에 따라 존재하지 않을 수 있는
    functional_tensor 서브모듈에서 직접 import를 시도합니다.
    실제 함수들은 torchvision.transforms.functional에 동일하게 존재하므로
    해당 모듈을 가리키는 shim(가상 모듈)을 sys.modules에 삽입합니다.
    """
    # 이미 로드된 경우 스킵
    if 'torchvision.transforms.functional_tensor' in sys.modules:
        return

    # 실제로 존재하는 경우에도 스킵
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        pass

    # torchvision.transforms.functional에서 동일한 심볼 가져와 shim 생성
    import torchvision.transforms.functional as _F

    mod = types.ModuleType('torchvision.transforms.functional_tensor')

    # basicsr 내부에서 자주 참조되는 함수 목록
    _symbols = (
        'rgb_to_grayscale',
        'adjust_brightness',
        'adjust_contrast',
        'adjust_saturation',
        'adjust_hue',
        'adjust_gamma',
        'adjust_sharpness',
        'normalize',
        'resize',
        'pad',
        'crop',
        'hflip',
        'vflip',
        'rotate',
        'perspective',
        'to_grayscale',
        'gaussian_blur',
        'invert',
        'posterize',
        'solarize',
        'autocontrast',
        'equalize',
    )

    for name in _symbols:
        if hasattr(_F, name):
            setattr(mod, name, getattr(_F, name))

    sys.modules['torchvision.transforms.functional_tensor'] = mod
    print(
        '[basicsr_compat] torchvision.transforms.functional_tensor shim 주입 완료.\n'
        f'                 torchvision 버전 호환 처리됨.'
    )
