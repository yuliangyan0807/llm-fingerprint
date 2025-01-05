import random
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from model_list import *
from metrics import *
from generation import *
from tqdm import tqdm
from typing import List
from utils import *
from datasets import load_from_disk, Dataset
import warnings
from itertools import combinations
from transformers import set_seed
import os

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

def optimize_prompts_with_pair_sampling( 
                                        model_list, 
                                        M=10, 
                                        sample_size=100, 
                                        alpha=1.0, 
                                        beta=1.0, 
                                        ):
    """
    Finds M optimized prompts from trigger_set by maximizing intra-model similarity and inter-model divergence with pair sampling.

    Parameters:
    - models: List of models.
    - M: Number of optimized prompts to find.
    - sample_size: Number of prompts to sample for evaluation.
    - alpha: Weight for intra-model similarity.
    - beta: Weight for inter-model divergence.
    - subset_size: Number of model pairs to sample for inter-model divergence.

    Returns:
    - List of optimized prompts.
    """
    print(f"loading dataset...")
    trigger_set = load_from_disk('./data/optimized_trigger')
    original_trigger_set = load_from_disk('./data/seed_trigger_set')
    # get the number of models
    model_number = len(model_list)
    prompt_number = len(trigger_set) // model_number
    # For recording the trigger's importance socre
    prompt_weights = []
    
    # iterate each prompt
    print(f"start to initial search")
    for i in tqdm(range(prompt_number)):
        # given the prompt, select the corresponding output of each model
        select_indices = [k * prompt_number + i for k in range(model_number)]
        # base models are orginal models, so an instruction tuning version model is also a base model here.
        base_model_indices = range(0, model_number, 2)
        assert len(base_model_indices) * 2 == model_number, "length error!"
        
        # Compute the intra similarity score
        intra_sim_score = []
        for j in base_model_indices:
            tokens1 = trigger_set[select_indices[j]]['tokens']
            tokens2 = trigger_set[select_indices[j + 1]]['tokens']
            intra_sim_score += [compute_intra_model_similarity(tokens1=tokens1, tokens2=tokens2)]    
        assert len(intra_sim_score) == model_number // 2, "length error!"
        
        inter_div_score = []
        # filter out the model and its fine tuning version pari
        forbidden_pairs = [(i, i + 1) for i in base_model_indices]
        model_pairs = [pair for pair in combinations(range(model_number), r=2) if pair not in forbidden_pairs]
        # Iterate all the possible pairs
        for pair in model_pairs:
            tokens1 = trigger_set[select_indices[pair[0]]]['tokens']
            tokens2 = trigger_set[select_indices[pair[1]]]['tokens']
            inter_div_score += [compute_inter_model_divergence(tokens1=tokens1, tokens2=tokens2)]

        intra_score = sum(intra_sim_score) / len(intra_sim_score)
        inter_score = sum(inter_div_score) / len(inter_div_score)
        
        prompt_weights.append(max(alpha * intra_score - beta * inter_score, 0))

    # Normalize weights to form a probability distribution
    total_weight = sum(prompt_weights)
    prompt_probs = [weight / total_weight for weight in prompt_weights]
    assert len(prompt_probs) == prompt_number, "length error!"

    # Sample prompts indices based on their importance weights
    candidate_prompts_indices = random.choices(range(prompt_number), 
                                              weights=prompt_probs, 
                                              k=sample_size)
    candidate_prompts_indices = list(set(candidate_prompts_indices))
    prompt_scores = []

    print(f"start to search the final trigger set")
    for i in tqdm(candidate_prompts_indices):
        # given the prompt, select the corresponding output of each model
        select_indices = [k * prompt_number + i for k in range(model_number)]
        # base models are orginal models, so an instruction tuning version model is also a base model here.
        base_model_indices = range(0, model_number, 2)
        
        # Compute the intra similarity score
        intra_sim_score = []
        for j in base_model_indices:
            entropy1 = trigger_set[select_indices[j]]['mean_entropy']
            entropy2 = trigger_set[select_indices[j + 1]]['mean_entropy']
            intra_sim_score += [abs(entropy1 - entropy2)]
        
        inter_div_score = []
        # filter out the model and its fine tuning version pari
        forbidden_pairs = [(i, i + 1) for i in base_model_indices]
        model_pairs = [pair for pair in combinations(range(model_number), r=2) if pair not in forbidden_pairs]
        # Iterate all the possible pairs
        for pair in model_pairs:
            entropy1 = trigger_set[select_indices[j]]['mean_entropy']
            entropy2 = trigger_set[select_indices[j + 1]]['mean_entropy']
            inter_div_score += [abs(entropy1 - entropy2)]

        intra_score = sum(intra_sim_score) / len(intra_sim_score)
        inter_socre = sum(inter_div_score) / len(inter_div_score)
        # Based on the entropy, we expect the intra score is low.
        score = beta * inter_socre - alpha * intra_score
        prompt_scores.append((i, score))
        
    assert len(prompt_scores) == len(candidate_prompts_indices)
    
    # Sort and select the top M prompts
    optimized_prompts_indices = sorted(prompt_scores, key=lambda x: x[1], reverse=True)[:M]
    indices = [i for i, score in optimized_prompts_indices]
    prompts = original_trigger_set.select(indices)['prompt']
    data = {"prompt" : prompts}
    data = Dataset.from_dict(data)
    
    data.save_to_disk('./data/final_trigger_set')
    
    return data

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