from datasets import load_dataset
from evaluate import load

# ds = load_dataset("tatsu-lab/alpaca")
# print(ds['corpus'])

# bertscore = load("bertscore")
# predictions = ["hello there"]
# references = ["hello here"]
# results = bertscore.compute(predictions=predictions, references=references, lang="en")

# print(results['f1'])

# input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
# perplexity = load("perplexity", module_type="metric")
# results = perplexity.compute(predictions=input_texts, model_id="/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/")
# print(results)

# ds = load_dataset("openai/gsm8k", "main")
# print(ds['train'][0])

ds = load_dataset("Intel/orca_dpo_pairs")