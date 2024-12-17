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

# MODEL_LABEL = ["llama3",
#                "llama3-ft",
#                "llama3-instruct",
#                "llama3-instruct-ft",
#                "llama3.1",
#                "llama3.1-ft",
#                "llama3.1-instruct",
#                "llama3.1-instruct-ft",
#                "mistral",
#                "mistral-ft",
#                "deepseek",
#                "deepseek-ft",
#                "deepseek-chat",
#                "deepseek-chat-ft",
#                "deepseek-math",
#                "deepseek-math-ft",
#                "qwen2.5",
#                "qwen2.5-ft",
#                ]

# LABEL = {"/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/" : 0,
#         "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_8_test" : 0,
#         "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/" : 1,
#         "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/" : 2,
#         "/home/yuliangyan/Code/llm-fingerprinting/llama3_1_8_ft_tiny_test" : 2,
#         "/home/yuliangyan/Code/llm-fingerprinting/llama3_1_8_ft_super_tiny_test" : 2,
#         "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_1_8_test" : 2,
#         "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/" : 3,
#         "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/" : 4,
#         "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_mistral_test" : 4,
#         "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/" : 5,
#         "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_deepseek_8_test" : 5,
#         "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/" : 6,
#         "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/" : 7,
#         "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/" : 8,
#         "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct" : 9,
#         }

MODEL_LIST = [
  # Llama 3.1 Family.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3-8B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-SuperNova-Lite', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-ARC-Potpourri-Induction-8B', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Unsloth-Meta-Llama-3.1-8B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-8B-Instruct-FP8', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/llama_3.1_8b_prop_logic_ft', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/calme-2.3-legalkit-8b', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Skywork', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/meta-llama_Llama-3.1-8B-Instruct-auto_gptq-int4-gs128-asym', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/SummLlama3.1-8B',
  # Qwen2.5 Family.
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-UMLS-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Math-IIO-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/FinancialAdvice-Qwen2.5-7B', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-AWQ', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-Uncensored', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-GPTQ-Int4', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-abliterated-v2', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/cybertron-v4-qw7B-UNAMGS', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/qwen2.5-7B-instruct-simpo', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/T.E-8.1',
  # Mistral Family.
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-base-instruct', 
  '/mnt/data/hf_models/mistral-7b-instruct/mistral_docs_sum_p1_full', 
  '/mnt/data/hf_models/mistral-7b-instruct/mistralai-Code-Instruct-Finetune-SG1-V5', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1-AWQ', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1_asm_60e4dc58', 
  '/mnt/data/hf_models/mistral-7b-instruct/original_glue_boolq', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1-GPTQ', 
  '/mnt/data/hf_models/mistral-7b-instruct/WeniGPT-Mistral-7B-instructBase', 
  '/mnt/data/hf_models/mistral-7b-instruct/full_v2_astromistral', 
  '/mnt/data/hf_models/mistral-7b-instruct/finetuned-mistral-7b',
]

# Model list for training the fingerprint extractor.
MODEL_LIST_TRAIN = [
  # Llama 3.1 Family.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3-8B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-SuperNova-Lite', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-ARC-Potpourri-Induction-8B', 
  # Maybe has some bugs.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-3.1-8B-Instruct-FP8', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/llama_3.1_8b_prop_logic_ft', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/calme-2.3-legalkit-8b', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Skywork', 
  # Maybe has some bugs.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/meta-llama_Llama-3.1-8B-Instruct-auto_gptq-int4-gs128-asym', 
  # Qwen2.5 Family.
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen-UMLS-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Math-IIO-7B-Instruct', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/FinancialAdvice-Qwen2.5-7B', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-AWQ', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/cybertron-v4-qw7B-UNAMGS', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/qwen2.5-7B-instruct-simpo', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/T.E-8.1',
  # Mistral Family.
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-base-instruct', 
  '/mnt/data/hf_models/mistral-7b-instruct/mistral_docs_sum_p1_full', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1_asm_60e4dc58', 
  '/mnt/data/hf_models/mistral-7b-instruct/original_glue_boolq', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1-GPTQ', 
  '/mnt/data/hf_models/mistral-7b-instruct/WeniGPT-Mistral-7B-instructBase', 
  '/mnt/data/hf_models/mistral-7b-instruct/finetuned-mistral-7b',
]

# Model list for evaluation the fingerprint extractor.
MODEL_LIST_TEST = [
  # Llama Family.
  '/mnt/data/hf_models/llama-3.1-8b-instruct/SummLlama3.1-8B',
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Unsloth-Meta-Llama-3.1-8B-Instruct', 
  '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8',
  # Qwen Family.
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-Uncensored', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-abliterated-v2', 
  '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct-GPTQ-Int4', 
  # Mistral Family.
  '/mnt/data/hf_models/mistral-7b-instruct/full_v2_astromistral', 
  '/mnt/data/hf_models/mistral-7b-instruct/mistralai-Code-Instruct-Finetune-SG1-V5', 
  '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1-AWQ',
]