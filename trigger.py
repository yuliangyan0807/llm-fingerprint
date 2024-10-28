import random
from transformers import set_seed
from datasets import load_dataset, Dataset, load_from_disk
from transformers import set_seed
from tqdm import tqdm
from model_list import *
from generation import *

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
    labels = []
    for i in range(len(DATASET)):
        labels += [i] * k
    data = {"prompt": data,
            "label": labels}
    data = Dataset.from_dict(data)
    data.save_to_disk("seed_trigger_set")
    
def construct_classifier_dataset(seed_dataset,
                                 sample_number_per_step: int,):
    model_labels = []
    outputs = []
    model_numbers = len(MODEL_LIST)
    
    for i in tqdm(range(len(seed_dataset))):
        data = seed_dataset[i]
        prompt = data['prompt']
        indices = random.sample(range(model_numbers), k=sample_number_per_step)
        for index in indices:
            model_name_or_path = MODEL_LIST[index]
            fine_tuned = False
            if "test" in model_name_or_path:
                fine_tuned = True
            try:
                output = generation(model_name_or_path=model_name_or_path,
                                    prompt=prompt,
                                    fine_tuned=fine_tuned)
                outputs.append(output)
                model_labels.append(LABEL[model_name_or_path])
            except RuntimeError as e:
                pass
            
    
    data = {'output': outputs,
            "model_label": model_labels}
    data = Dataset.from_dict(data)
    data.save_to_disk("classifier_dataset")
        

if __name__ == "__main__":
    # construct_seed_trigger_set(100)
    trigger_set = load_from_disk("data/seed_trigger_set")
    # print(trigger_set[310])
    construct_classifier_dataset(trigger_set, sample_number_per_step=2)