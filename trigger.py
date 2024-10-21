import random
from transformers import set_seed
from datasets import load_dataset, Dataset, load_from_disk
from transformers import set_seed
from tqdm import tqdm

# Math dataset
GSM8K = load_dataset("openai/gsm8k", "main")['train']['question'] # question
MATHINSTRUCT = load_dataset("TIGER-Lab/MathInstruct")['train']['instruction'] # instruction
# Safety dataset
HARMFULDATASET = load_dataset("LLM-LAT/harmful-dataset")['train']['prompt'] # prompt
ADVBENCH = load_dataset("walledai/AdvBench")['train']['prompt'] # prompt
# Commonsense dataset
COMMONSENSECANDIDATES = load_dataset("commonsense-index-dev/commonsense-candidates", "all")['train']['task'] # task
COMMONSENSEQA = load_dataset("tau/commonsense_qa")['train']['question'] # question

DATASET = [GSM8K, MATHINSTRUCT, HARMFULDATASET, ADVBENCH, COMMONSENSECANDIDATES, COMMONSENSEQA]

set_seed(42)

def construct_seed_trigger_set(k: int):
    data = []
    for dataset in tqdm(DATASET):
        # random_indices = random.sample(range(len(dataset)), k)
        # subset = dataset.select(random_indices)
        subset = random.choices(dataset, k=k)
        for prompt in tqdm(subset):
            data.append(prompt)
    data = {"prompt": data}
    data = Dataset.from_dict(data)
    data.save_to_disk("seed_trigger_set")

if __name__ == "__main__":
    # construct_seed_trigger_set(100)
    trigger_set = load_from_disk("seed_trigger_set")
    print(trigger_set[110])