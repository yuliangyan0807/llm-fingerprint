# Model list for training the fingerprint extractor.
MODEL_LIST_TRAIN = [
  # Llama 3.1 Family. 
  # TODO
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-2-13b-chat-hf',
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Numerical_Reasoning_llama2',
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-14B-Instruct-1M',
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/T3Q-qwen2.5-14b-v1.0-e3',
  '/mnt/data/hf_models/mistral-7b-instruct/Trinity-2-Codestral-22B',
  '/mnt/data/hf_models/mistral-7b-instruct/Trinity-2-Codestral-22B-v0.2',
  # From here.
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-8B-UltraMedical', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-ARC-Potpourri-Induction-8B',
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-8bit-Instruct-sql-v3', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-3.1-8b-instruct-ultrafeedback-single-judge', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-SuperNova-Lite', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/task-1-meta-llama-Meta-Llama-3.1-8B-Instruct-1736201342',
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/llama_3.1_8b_prop_logic_ft', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/prm800k_llama_lora', 
  # '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-3_1-8b-instruct-fake-news',
  # Qwen2.5 Family.
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-UMLS-7B-Instruct', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Human-Like-Qwen2.5-7B-Instruct',
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/cybertron-v4-qw7B-UNAMGS', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/qwen2.5-7B-instruct-simpo', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-Uncensored',
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Math-IIO-7B-Instruct', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/T.E-8.1',
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/FinancialAdvice-Qwen2.5-7B', 
  # '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-Rui-SE',
  # Mistral Family.
  # '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
  # '/mnt/data/hf_models/mistral-7b-instruct/radia-fine-tune-mistral-7b-lora',
  # '/mnt/data/hf_models/mistral-7b-instruct/astromistralv2', 
  # '/mnt/data/hf_models/mistral-7b-instruct/mistralai-Code-Instruct-Finetune-SG1-V5', 
  # '/mnt/data/hf_models/mistral-7b-instruct/mistral_instruct_generation',
  # '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1_asm_60e4dc58', 
  # '/mnt/data/hf_models/mistral-7b-instruct/original_glue_boolq', 
  # '/mnt/data/hf_models/mistral-7b-instruct/WeniGPT-Mistral-7B-instructBase', 
  # '/mnt/data/hf_models/mistral-7b-instruct/finetuned-mistral-7b',
  # '/mnt/data/hf_models/mistral-7b-instruct/full_v2_astromistral', 
]

MODEL_LIST_UNSEEN = [
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct',
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-Doctor-3.2-3B-Instruct', 
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-Sentient-3.2-3B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
]