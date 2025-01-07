# This is model list for our pre-experiments.
# MODEL_LIST = [
#               "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-ft",
#               "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-instruct-ft",
#               "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama31-ft",
#               "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-instruct-ft",
#               "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/mistral-ft",
#               "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-ft",
#               "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-chat-ft",
#               "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-math-instruct-ft",
#               "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/",
#               "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/qwen25-ft",
#               ]

# Model list for training the fingerprint extractor.
MODEL_LIST_TRAIN = [
  # Llama 3.1 Family.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-8B-UltraMedical', # work
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Unsloth-Meta-Llama-3.1-8B-Instruct', # work
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-SuperNova-Lite', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-ARC-Potpourri-Induction-8B', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-8B-Instruct-FP8', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/llama_3.1_8b_prop_logic_ft', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-3.1-8b-instruct-ultrafeedback-single-judge', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Skywork', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8', 
  # Qwen2.5 Family.
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-UMLS-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Human-Like-Qwen2.5-7B-Instruct',
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Math-IIO-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/FinancialAdvice-Qwen2.5-7B', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/cybertron-v4-qw7B-UNAMGS', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/qwen2.5-7B-instruct-simpo', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-Uncensored',
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/T.E-8.1',
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-Rui-SE',
  # Mistral Family.
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
  '/mnt/data/hf_models/mistral-7b-instruct/astromistralv2', 
  '/mnt/data/hf_models/mistral-7b-instruct/mistralai-Code-Instruct-Finetune-SG1-V5', # work
  '/mnt/data/hf_models/mistral-7b-instruct/full_v2_astromistral', # work
  '/mnt/data/hf_models/mistral-7b-instruct/mistral_instruct_generation',
  '/mnt/data/hf_models/mistral-7b-instruct/radia-fine-tune-mistral-7b-lora',
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1_asm_60e4dc58', 
  '/mnt/data/hf_models/mistral-7b-instruct/original_glue_boolq', 
  '/mnt/data/hf_models/mistral-7b-instruct/WeniGPT-Mistral-7B-instructBase', 
  '/mnt/data/hf_models/mistral-7b-instruct/finetuned-mistral-7b',
]

MODEL_LIST_UNSEEN = [
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct',
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-Doctor-3.2-3B-Instruct', # work
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-Sentient-3.2-3B-Instruct', # work
  '/mnt/data/hf_models/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct-bnb-4bit', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
]