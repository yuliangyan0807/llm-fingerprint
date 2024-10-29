import random
import numpy as np
from datasets import load_dataset, Dataset
from model_list import *
from metrics import *
from generation import *

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
    for prompt in trigger_set:
        intra_sim_score = sum(
            compute_intra_model_similarity(model, fine_tuned, prompt) 
            for model, fine_tuned in zip(models, fine_tuned_models)
        )
        
        # Randomly sample subset_size pairs of models for inter-model divergence
        model_pairs = random.sample([(i, j) for i in range(len(models)) for j in range(i + 1, len(models))], subset_size)
        inter_div_score = sum(
            compute_inter_model_divergence(models[i], models[j], prompt) 
            for i, j in model_pairs
        )
        
        # Importance weight based on prompt characteristics
        weight = alpha * intra_sim_score - beta * inter_div_score
        prompt_weights.append(weight)

    # Normalize weights to form a probability distribution
    total_weight = sum(prompt_weights)
    prompt_probs = [weight / total_weight for weight in prompt_weights]

    # Sample prompts based on their importance weights
    candidate_prompts = random.choices(trigger_set, weights=prompt_probs, k=sample_size)
    prompt_scores = []

    for prompt in candidate_prompts:
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
    seed_trigger_set = load_dataset("./data/seed_trigger_set")
    models = [
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B-Instruct/",
        "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/",
        # "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B-Instruct/",
        "/mnt/data/yuliangyan/mistralai/Mistral-7B-v0.1/",
        "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/",
        "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-chat/",
        # "/mnt/data/yuliangyan/deepseek-ai/deepseek-math-7b-instruct/",
        # "/mnt/data/yuliangyan/Qwen/Qwen2.5-7B/",
        # "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct",
    ]
    fine_tuned_models = [
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-ft",
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-instruct-ft",
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama31-ft",
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/mistral-ft",
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-ft",
        "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/deepseek-chat-ft",
    ]

    # Find optimized prompts
    optimized_prompts = optimize_prompts_with_pair_sampling(seed_trigger_set, 
                                                            models, 
                                                            fine_tuned_models, 
                                                            M=15, 
                                                            sample_size=150, 
                                                            alpha=0.6, 
                                                            beta=0.4
                                                            )
    print("Optimized prompts:", optimized_prompts)