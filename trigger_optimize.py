import random
import numpy as np

# Hypothetical metric functions
def compute_intra_model_similarity(model, fine_tuned_model, prompt):
    """Calculates similarity between model and fine-tuned model outputs for a given prompt."""
    # Placeholder for real implementation
    return np.random.rand()

def compute_inter_model_divergence(model1, model2, prompt):
    """Calculates divergence between two different models' outputs for a given prompt."""
    # Placeholder for real implementation
    return np.random.rand()

def optimize_prompts_with_pair_sampling(trigger_set, models, fine_tuned_models, M=10, sample_size=100, alpha=1.0, beta=1.0, subset_size=5):
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
        weight = alpha * intra_sim_score + beta * inter_div_score
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
    
    return [prompt for prompt, score in optimized_prompts]

# Example usage
trigger_set = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6", "prompt7", "prompt8", "prompt9", "prompt10"]
models = ["model1", "model2", "model3"]
fine_tuned_models = ["model1_fine", "model2_fine", "model3_fine"]

# Find optimized prompts
optimized_prompts = optimize_prompts_with_pair_sampling(trigger_set, models, fine_tuned_models, M=5, sample_size=5, alpha=1.0, beta=1.0)
print("Optimized prompts:", optimized_prompts)