MODEL_LIST = [
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-ft",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-instruct-ft",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama31-ft",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-instruct-ft",
              "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/mistral-ft",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-ft",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-chat-ft",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-math-instruct-ft",
              "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/qwen25-ft",
              # "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
            #   "/mnt/data/yuliangyan/google/gemma-2-2b",
            #   "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/gemma2-ft",
              ]

MODEL_LABEL = ["llama3",
               "llama3-ft",
               "llama3-instruct",
               "llama3-instruct-ft",
               "llama3.1",
               "llama3.1-ft",
               "llama3.1-instruct",
               "llama3.1-instruct-ft",
               "mistral",
               "mistral-ft",
               "deepseek",
               "deepseek-ft",
               "deepseek-chat",
               "deepseek-chat-ft",
               "deepseek-math",
               "deepseek-math-ft",
               "qwen2.5",
               "qwen2.5-ft",
              #  "phi3",
              #  "gemma2",
              #  "gemma2-ft"
               ]
# TODO
# gemmma-2

LABEL = {"/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/" : 0,
        "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_8_test" : 0,
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/" : 1,
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/" : 2,
        "/home/yuliangyan/Code/llm-fingerprinting/llama3_1_8_ft_tiny_test" : 2,
        "/home/yuliangyan/Code/llm-fingerprinting/llama3_1_8_ft_super_tiny_test" : 2,
        "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_1_8_test" : 2,
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/" : 3,
        "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/" : 4,
        "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_mistral_test" : 4,
        "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/" : 5,
        "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_deepseek_8_test" : 5,
        "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/" : 6,
        "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/" : 7,
        "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/" : 8,
        "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct" : 9,
        }