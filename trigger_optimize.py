from datasets import load_dataset, Dataset, load_from_disk
from model_list import *
from metrics import *
from generation import *
from tqdm import tqdm
from typing import List
from utils import *
from datasets import load_from_disk, Dataset
from transformers import T5EncoderModel
import warnings
from transformers import set_seed

def pre_generate_trajectory(
    model_list: List[str],
    # seed_trigger_set,
    batch_size: int=50,
    max_new_tokens: int=64,
):
    """
    Given the trigger set, pre generate the trajectory of the models in the model lists.
    Tajectory include tokens, token_probs, entropy, and so on.
    """
    print(f"loading dataset...")
    seed_trigger_set = load_from_disk('./data/seed_trigger_set')
    print(f"{len(seed_trigger_set)} data in total")
    prompts = seed_trigger_set['prompt']
    
    assert len(seed_trigger_set) % batch_size == 0, "choose another batch size!"
    
    tokens_, token_probs_, decode_output_, entropy_, mean_entropy_, labels = [], [], [], [], [], []
    
    # Iterate the model list
    for label, model_name_or_path in enumerate(model_list):
        # load the model
        model, tokenizer = load_hf_model(model_name_or_path)
        # iterate the prompt
        for i in tqdm(range(0, len(seed_trigger_set), batch_size)):
            batch_prompt = prompts[i : i + batch_size]
            tokens, token_probs, decode_output, entropy, mean_entropy = batch_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=batch_prompt,
                max_new_tokens=max_new_tokens
            )
            tokens_ = tokens_ + tokens
            token_probs_ = token_probs_ + token_probs
            decode_output_ = decode_output_ + decode_output
            entropy_ = entropy_ + entropy
            mean_entropy_ = mean_entropy_ + mean_entropy
    
    print(f"saving dataset...")
    optimized_trigger_set = {"tokens": tokens_,
                             "token_probs": token_probs_,
                             "decode_output": decode_output_,
                             "entropy": entropy_,
                             "mean_entropy": mean_entropy_,
                            #  "labels": labels
                             }
    optimized_trigger_set = Dataset.from_dict(optimized_trigger_set)
    optimized_trigger_set.save_to_disk('./data/trajectory_set_unseen')

def optimize_triggers(
    trajectory_set, 
    model_list: List[str], 
    family_name: str, 
    tokenizer, 
    model_path: str, 
    fold: int, 
    threshold: int, 
) -> Dataset:
    """
    Finds M optimized prompts from trigger_set by maximizing intra-model similarity and inter-model divergence with pair sampling.

    Parameters:
    - trajectory_set: pre-collected trajectories towards the prompt. 
    - model_list: List of models. 
    - family_name: victim model family. 
    - fold: current fold to be optimized. 

    Returns:
    - List of optimized prompts.
    """
    assert family_name in ['llama', 'mistral', 'qwen'], "Out of the family pre-defined!"
    
    print(f"loading dataset...")
    original_trigger_set = load_from_disk('./data/seed_trigger_set')
    prompt_number = len(original_trigger_set)
    assert prompt_number == 600, "error!"
    
    print(f"Loading the extractor...")
    model = T5EncoderModel.from_pretrained(
        model_path, 
        output_hidden_states=True,
        ignore_mismatched_sizes=True, 
    )

    # Each fold has a subset of the trigger set. 
    res, model_list_train, model_list_eval = get_cross_validation_datasets(
        trajectory_set=trajectory_set, 
        model_list=model_list, 
        fold=fold
    )
    
    # Specify the target family. 
    model_number_per_family = len(trajectory_set) // 3
    if family_name == 'llama':
        target_index = 0
    elif family_name == 'qwen':
        target_index = 7
    else:
        target_index = 14
    
    ideal_value = model_number_per_family - 1
        
    contrastive_dataset = construct_contrastive_dataset(
        tokenizer=tokenizer, 
        raw_data=res['train'], 
        model_list=model_list_train, 
        select_trigger=True, 
    )
    
    # We only need to save the index of the prompt. 
    optimized_trigger_set = []    
    # iterate each prompt
    print(f"Start to search...")
    for i, sample in enumerate(contrastive_dataset):
        batch_input_ids = sample['input_ids']
        batch_attention_mask = sample['attention_mask']
        inputs = {
            'input_ids' : torch.tensor(batch_input_ids),
            'attention_mask' : torch.tensor(batch_attention_mask),
        }
        last_hidden_states = model(**inputs).last_hidden_state
        aggregated_hidden_states = torch.sum(last_hidden_states, dim=-1)
        aggregated_hidden_states = F.normalize(aggregated_hidden_states, dim=-1)
        simlarity_marix = torch.matmul(aggregated_hidden_states, aggregated_hidden_states.T)
        # simlarity_marices.append(simlarity_marix)
        
        target_logits = torch.argmax(simlarity_marix[target_index + 1: target_index + model_number_per_family, :], dim=1)
        # Count the number of the suspect models has the most similary with the victim model. 
        val = torch.sum(target_logits == target_index).item()
        if val >= threshold:
            optimized_trigger_set.append(i)
            print(original_trigger_set[i])
    
    dataset = {
        'prompt_index' : optimized_trigger_set
    }
    dataset = Dataset.from_dict(dataset)
    save_dir = f"./data/optimized_trigger_set_{family_name}_{fold}"
    dataset.save_to_disk(save_dir)
    
    return dataset

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # model_list = MODEL_LIST_TRAIN
    model_list = MODEL_LIST_UNSEEN

    # Find optimized prompts
    # optimized_prompts = optimize_prompts_with_pair_sampling(
    #                                                     model_list=model_list, 
    #                                                     M=64, 
    #                                                     sample_size=200, 
    #                                                     alpha=0.3, 
    #                                                     beta=0.7,
    #                                                     )
    
    # data = load_from_disk('./data/seed_trigger_set')
    # print(len(data))
    
    # Collect the trajectories of the models.
    pre_generate_trajectory(model_list=model_list,
                            batch_size=50
                            )