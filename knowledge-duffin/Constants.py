import os
max_model_length = 4096
max_new_tokens = 1024
MaskIds = []
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"