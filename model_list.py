MODEL_LIST = ["/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",
              "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_8_test",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3_1_8_ft_tiny_test",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3_1_8_ft_super_tiny_test",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/stanford_alpaca/saved_models_llama3_1_8_test",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/",
              "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/",
              "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/instruction_tuning_models/saved_models_mistral_test",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/",
              "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/stanford_alpaca/saved_models_deepseek_8_test",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/",
              "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/",
              "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
              ]

MODEL_LABEL = ["llama3",
               "llama3-ft",
               "llma3-instruct",
               "llama3.1",
               "llama3.1-tiny",
               "llama3.1-super-tiny",
               "llama3.1-ft",
               "llama3.1-instruct",
               "mistral",
               "mistral-ft",
               "deepseek",
               "deepseek-ft",
               "deepseek-chat",
               "deepseek-math",
               "qwen2.5",
               "phi3"]
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