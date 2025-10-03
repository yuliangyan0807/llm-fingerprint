import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM

def load_hf_model_TF(model_name_or_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load Hugging Face Transformer model."""
    print(f"Loading model '{model_name_or_path}' on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, return_dict=True, device_map="auto",
            output_hidden_states=True, local_files_only=True, load_in_4bit=True,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, padding_side='left', local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_hf_model_VLM(model_name, is_quantization=False):
    """Load VLM model with optional quantization."""
    print(f"Loading VLM model: {model_name}")
    llm = LLM(
        model=model_name, gpu_memory_utilization=0.9,
        tensor_parallel_size=4, max_model_len=4096,
        distributed_executor_backend='mp', disable_custom_all_reduce=True,
        trust_remote_code=False, enable_lora=True, max_lora_rank=64
    ) if not is_quantization else LLM(
        model=model_name, gpu_memory_utilization=0.91,
        tensor_parallel_size=1, max_model_len=4096,
        trust_remote_code=True, quantization="bitsandbytes", load_format="bitsandbytes"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer
