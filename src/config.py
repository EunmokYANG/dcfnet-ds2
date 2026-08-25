"""전역 설정. 경로와 데이터셋 스펙을 한 곳에서 관리한다."""

from pathlib import Path

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"        # 원본 대용량 CSV (선택)
RAW_DIR = DATA_DIR / "raw"              # 클래스당 <=10k 샘플링 결과
PROCESSED_DIR = DATA_DIR / "processed"  # npz (train/test 분리 완료)

OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for _d in (SOURCE_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 공통
SEED = 1
TEST_SIZE = 0.3
VAL_SIZE = 0.2          # train 내부에서 다시 분리
MAX_PER_CLASS = 10_000  # 통제 샘플링 상한 (0 이면 샘플링 없이 전체 사용)
SCALER = "signedlog"    # minmax | standard | robust | signedlog |
                        # signedlog_minmax | none
SPLIT = "random"        # random | temporal

# ---------------------------------------------------------------- 데이터셋
DATASETS = {
    "ciciot2023": {
        "raw_file": "CICIoT2023.csv",
        "label_col": "label",
        # 선행 연구와 동일: 상수 컬럼만 제거. 식별자 피처 없음.
        "drop_cols": [],
        # CICIoT2023 에는 명목형 코드나 비트마스크가 없다.
        # 이진 플래그(fin_flag_number 등)는 이미 컬럼별로 분리되어 있다.
        "nominal_cols": [],
        "bitmask_cols": [],
        "category_map": "CICIOT_CATEGORY_MAPPING",
        "time_col": None,          # 타임스탬프 없음 -> 시간 분할 불가
        "classes": [
            "Benign", "BruteForce", "DDoS", "DoS",
            "Mirai", "Recon", "Spoofing", "Web",
        ],
    },
    "nfunsw": {
        "raw_file": "NF-UNSW-NB15-v3.csv",
        "label_col": "Attack",
        # 타임스탬프 + 이진라벨 + 네트워크 식별자를 모두 제거.
        # 식별자를 남기면 차수 게이트가 호스트 암기를 반영하게 되어
        # alpha_k 해석이 성립하지 않는다.
        "drop_cols": [
            "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label",
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
        ],
        # 수치적 순서가 의미 없는 컬럼.
        #   nominal : 프로토콜 코드. 6 > 1 이라는 순서에 의미가 없다.
        #   bitmask : TCP 플래그 비트마스크. 24 는 8+16 이지 23+1 이 아니다.
        "nominal_cols": ["PROTOCOL", "L7_PROTO"],
        "bitmask_cols": ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"],
        "category_map": None,
        "time_col": "FLOW_START_MILLISECONDS",
        "classes": [
            "Analysis", "Backdoor", "Benign", "DoS", "Exploits",
            "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
        ],
    },
}


# ---------------------------------------------------------------- 라벨 매핑
# CICIoT2023 원본은 34개 세부 라벨이므로 8개 카테고리로 통합한다.
# NF-UNSW-NB15-v3 는 Attack 컬럼이 이미 10개 카테고리라 매핑이 없다.
CICIOT_CATEGORY_MAPPING = {
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS", "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",

    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS", "DoS-HTTP_Flood": "DoS",

    "Mirai-greeth_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Mirai-greip_flood": "Mirai",

    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon", "Recon-PingSweep": "Recon",
    "VulnerabilityScan": "Recon",

    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",

    "DictionaryBruteForce": "BruteForce",

    "SqlInjection": "Web", "BrowserHijacking": "Web",
    "CommandInjection": "Web", "XSS": "Web",
    "Backdoor_Malware": "Web", "Uploading_Attack": "Web",

    "BenignTraffic": "Benign",
}

# ---------------------------------------------------------------- 학습
# TF 2.13 미만(예: Windows 네이티브 GPU용 2.10)은 .keras 포맷을 지원하지 않는다.
# 그 경우 아래를 ".h5" 로 바꾸면 전 스크립트가 함께 따라간다.
MODEL_EXT = ".keras"

EPOCHS = 500
BATCH_SIZE = 1024   # 원본 전체(100만행+) 대응
LEARNING_RATE = 1e-3
FOCAL_GAMMA = 3.0
FOCAL_ALPHA = 0.75
MAX_DEGREE = 4
EARLY_STOP_PATIENCE = 20

# ---------------------------------------------------------------- 실험 변형
VARIANTS = {
    "m0": dict(degree_separated=False, gate_mode=None,
               desc="SHIELD-Net baseline (residual cross x3)"),
    "m1": dict(degree_separated=True, gate_mode="uniform",
               desc="degree-separated, uniform weights"),
    "m2": dict(degree_separated=True, gate_mode="global",
               desc="degree-separated + global gate"),
    "m3": dict(degree_separated=True, gate_mode="instance",
               desc="degree-separated + instance gate + MLP"),
    # --- 재설계 모델 (MLP 제거 + 차수 정규화 + 선형 readout) ---
    "m4": dict(kind="ds", degree_separated=True, gate_mode=None,
               desc="DCFNet-DS (proposed): linear readout, no MLP, no gate"),
    "m4g": dict(kind="ds", use_gate=True, degree_separated=True,
                gate_mode=None, desc="M4 + degree gate (ablation)"),
    "m4m": dict(kind="ds", use_mlp=True, degree_separated=True,
                gate_mode=None, desc="M4 + MLP branch (ablation)"),
    # --- 신경망 baseline ---
    "nn_mlp": dict(kind="mlp", degree_separated=False, gate_mode=None,
                   desc="MLP baseline (no cross network)"),
}
