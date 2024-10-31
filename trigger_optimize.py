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
        
        model_label = [[label] for _ in range(len(seed_trigger_set))]
        labels = labels + model_label
    
    print(f"saving dataset...")
    optimized_trigger_set = {"prompt": prompts,
                             "tokens": tokens_,
                             "token_probs": token_probs_,
                             "decode_output": decode_output_,
                             "entropy": entropy_,
                             "mean_entropy": mean_entropy_
                             }
    optimized_trigger_set = Dataset.from_dict(optimized_trigger_set)
    optimized_trigger_set.save_to_disk('./data/optimized_trigger_set')

# Hypothetical metric functions
def compute_intra_model_similarity(model, fine_tuned_model, prompt):
    """
    Calculates similarity between model and fine-tuned model outputs for a given prompt.
    """
    tokens1, token_probs1, text1 = generation(model, prompt)
    tokens2, token_probs2, text2 = generation(fine_tuned_model, prompt)
    
    res = jaccard_similarity(tokens1, tokens2)
    
    return res

def compute_inter_model_divergence(model1, model2, prompt):
    """Calculates divergence between two different models' outputs for a given prompt."""
    tokens1, token_probs1, text1 = generation(model1, prompt)
    tokens2, token_probs2, text2 = generation(model2, prompt)
    
    res = jaccard_similarity(tokens1, tokens2)
    
    return res

def optimize_prompts_with_pair_sampling(trigger_set, 
                                        models, 
                                        fine_tuned_models,
                                        M=10, 
                                        sample_size=100, 
                                        alpha=1.0, 
                                        beta=1.0, 
                                        subset_size=5):
    """
    Finds M optimized prompts from trigger_set by maximizing intra-model similarity and inter-model divergence with pair sampling.

    Parameters:
    - trigger_set: List of all prompts.
    - models: List of base models.
    - fine_tuned_models: Corresponding fine-tuned models.
    - M: Number of optimized prompts to find.
    - sample_size: Number of prompts to sample for evaluation.
    - alpha: Weight for intra-model similarity.
    - beta: Weight for inter-model divergence.
    - subset_size: Number of model pairs to sample for inter-model divergence.

    Returns:
    - List of optimized prompts.
    """
    prompt_weights = []
    
    # Calculate importance weights for each prompt
    print(f"start to initial search")
    for data in tqdm(trigger_set):
        # print(data)
        prompt = data['prompt']
        # intra_sim_score = sum(
        #     compute_intra_model_similarity(model, fine_tuned, prompt) 
        #     for model, fine_tuned in zip(models, fine_tuned_models)
        # )
        intra_sim_score = sum(
            compute_intra_model_similarity(model, fine_tuned, prompt) 
            for model, fine_tuned in zip(models, fine_tuned_models)
        ) / len(models)
        
        # Randomly sample subset_size pairs of models for inter-model divergence
        model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
        inter_div_score = sum(
            compute_inter_model_divergence(models[i], models[j], prompt) 
            for i, j in model_pairs
        ) / subset_size
        
        # Importance weight based on prompt characteristics
        weight = max(alpha * intra_sim_score - beta * inter_div_score, 0)
        prompt_weights.append(weight)

    # Normalize weights to form a probability distribution
    total_weight = sum(prompt_weights)
    prompt_probs = [weight / total_weight for weight in prompt_weights]

    # Sample prompts based on their importance weights
    candidate_prompts = random.choices(trigger_set, weights=prompt_probs, k=sample_size)
    prompt_scores = []

    print(f"start to search the final trigger set")
    for prompt in tqdm(candidate_prompts):
        intra_sim_score = 0
        inter_div_score = 0

        # Compute intra-model similarity
        for model, fine_tuned in zip(models, fine_tuned_models):
            intra_sim_score += compute_intra_model_similarity(model, fine_tuned, prompt)
        
        # Sample model pairs dynamically for inter-model divergence
        model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
        inter_div_score += sum(
            compute_inter_model_divergence(models[i], models[j], prompt) 
            for i, j in model_pairs
        )

        # Calculate overall score
        score = alpha * intra_sim_score - beta * inter_div_score
        prompt_scores.append((prompt, score))

    # Sort and select the top M prompts
    optimized_prompts = sorted(prompt_scores, key=lambda x: x[1], reverse=True)[:M]
    prompts = [prompt for prompt, score in optimized_prompts]
    
    data = {"prompt": prompts}
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk('./data/optimized_trigger_set')
    
    return prompts

if __name__ == '__main__':
    # Example usage
    # seed_trigger_set = load_from_disk("./data/seed_trigger_set")
    # print(len(seed_trigger_set))
    # seed_trigger_set = seed_trigger_set.select(range(0, 5))

    # # Find optimized prompts
    # optimized_prompts = optimize_prompts_with_pair_sampling(seed_trigger_set, 
    #                                                         models, 
    #                                                         fine_tuned_models, 
    #                                                         M=15, 
    #                                                         sample_size=150, 
    #                                                         alpha=0.4, 
    #                                                         beta=0.6,
    #                                                         subset_size=1
    #                                                         )
    # print("Optimized prompts:", optimized_prompts)
    
    model_list = MODEL_LIST
    pre_generate_trajectory(model_list=model_list,
                            # seed_trigger_set=seed_trigger_set,
                            batch_size=50,
                            max_new_tokens=16)