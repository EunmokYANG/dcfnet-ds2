"""환경 점검. 설치 직후 가장 먼저 실행한다.

실행:  python -m src.check_env
"""

import platform
import sys


def main():
    print("=" * 58)
    print(f"Python   : {sys.version.split()[0]}  ({platform.system()})")

    try:
        import numpy as np
        print(f"numpy    : {np.__version__}")
    except ImportError:
        print("numpy    : 미설치")
        np = None

    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow: 미설치 -> pip install -r requirements.txt")
        return

    tf_ver = tf.__version__
    print(f"tensorflow: {tf_ver}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU      : {len(gpus)}개 {[g.name for g in gpus]}")

    print("-" * 58)
    major, minor = (int(v) for v in tf_ver.split(".")[:2])

    if np is not None and int(np.__version__.split(".")[0]) >= 2 and (major, minor) < (2, 17):
        print("[경고] numpy 2.x 는 TF 2.17 미만과 호환되지 않습니다.")
        print("       pip install \"numpy<2.0\"")

    if (major, minor) < (2, 13):
        from src import config as C
        if C.MODEL_EXT != ".h5":
            print(f"[경고] TF {tf_ver} 는 .keras 포맷을 지원하지 않습니다.")
            print('       src/config.py 의 MODEL_EXT 를 ".h5" 로 바꾸세요.')

    if platform.system() == "Windows" and not gpus and (major, minor) > (2, 10):
        print("[안내] TF 2.11 이상은 네이티브 Windows 에서 GPU 를 지원하지 않습니다.")
        print("       CPU 로 진행해도 이 규모(약 4만 x 40)에서는 충분합니다.")
        print("       GPU 가 필요하면 WSL2 또는 TF 2.10 + CUDA 11.2 를 쓰세요.")

    if gpus:
        print("[확인] GPU 사용 가능.")
    print("=" * 58)

    # 실제 연산 한 번 돌려서 동작 확인
    import numpy as _np
    from src.model import build_model
    m = build_model(input_dim=40, num_classes=8, degree_separated=True)
    x = _np.random.normal(size=(4, 40)).astype("float32")
    p = m.predict(x, verbose=0)
    print(f"[확인] 모델 빌드/추론 정상. 출력 shape={p.shape}, "
          f"파라미터={m.count_params():,}")


if __name__ == "__main__":
    main()
