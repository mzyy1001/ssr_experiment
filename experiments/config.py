"""
Configuration for PS-SSR experiments.
Connects to chen server (8x A100-80GB) for Qwen3-8B inference.
"""
import os

# ─── Model paths (on chen server) ───
QWEN3_8B_PATH = "/data/chenhongrui/models/Qwen3-8B"

# Embedding model for SSR (Chinese medical domain)
EMBEDDING_MODEL = "iic/nlp_corom_sentence-embedding_chinese-base-medical"
# Fallback: general Chinese embedding
EMBEDDING_MODEL_FALLBACK = "BAAI/bge-base-zh-v1.5"

# ─── Server config ───
CHEN_SERVER = {
    "host": "122.225.39.134",
    "port": 2222,
    "user": "chenhongrui",
    "key": os.path.expanduser("~/.ssh/id_server_chen"),
}

# vLLM server (if running)
VLLM_BASE_URL = "http://127.0.0.1:8071/v1"
VLLM_API_KEY = "EMPTY"

# ─── Data paths (local) ───
DATA_DIR = "/home/mzyy1001/business/data"
RESULTS_DIR = "/home/mzyy1001/business/results"
FILTER_DF = f"{DATA_DIR}/1_filter_df.csv"
MEANINGFUL_DF = f"{DATA_DIR}/2_meaningful_df.csv"
CLUSTER_TOPICS = f"{DATA_DIR}/3_cluster_topics.json"
BASELINE_RESULTS = f"{RESULTS_DIR}/all_questions.json"

# ─── Experiment hyperparameters ───
# Persona vector extraction
EXTRACTION_LAYERS = [8, 12, 16, 20, 24]  # Qwen3-8B has 32 layers
N_SAMPLES_PER_CLUSTER = [10, 20, 50]
DEFAULT_N_SAMPLES = 20
DEFAULT_LAYER = 16  # middle layer, good default for steering

# Steering
STEERING_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_ALPHA = 2.0

# SSR
SSR_EPS = 0.1  # min-subtraction epsilon (from Project 1)

# Survey questions of interest
INTERESTED_QUESTIONS = {
    (8, 3), (8, 4),
    (9, 3), (9, 4), (9, 5),
    (11, 3),
}

# ─── Conda env on chen server ───
CONDA_ENV = "llama_qwen"  # has transformers 4.57.0
CONDA_ACTIVATE = f"source /data/anaconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV}"
