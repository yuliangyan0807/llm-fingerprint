import itertools

MODEL_LIST = ["/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/",
              "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/",
              "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/",
              "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/",
              "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/",
              "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
              ]
pairs = []
for model_pair in itertools.combinations(MODEL_LIST, 2):
    pairs.append(model_pair)
print(pairs[-1][0])