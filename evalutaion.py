from metrics import *
from generation import *
from typing import List
from utils import load_hf_model
from itertools import combinations
import numpy as np
from tqdm import tqdm
from model_list import *
from datasets import load_from_disk

def evaluate(
    trigger_set,
    model_list: List[str],
):
    pass
    # batch_size = len(trigger_set)
    num_models = len(model_list)
    prompts = trigger_set['prompt']
    model_tokens, model_token_probs, model_mean_entropy = [], [], []
    # Record the fingerprint towards the trigger set.
    for i in range(len(model_list)):
        model, tokenizer = load_hf_model(model_name_or_path=model_list[i])
        batch_tokens, token_probs, decoded_output, entropy, mean_entropy = batch_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompts,
        )
        model_tokens.append(batch_tokens)
        model_token_probs.append(token_probs)
        model_mean_entropy.append(mean_entropy)
        assert len(model_tokens) == len(model_token_probs), "length error!"
    
    edit_distance_matrix = np.zeros((num_models, num_models))
    jaccard_simlarity_matrix = np.zeros((num_models, num_models))
    entropy_simlarity_matrix = np.zeros((num_models, num_models))
    
    for i, j in tqdm(combinations(range(len(model_list)), r=2)):
        batch_tokens1 = model_tokens[i] # (batch_size, seq_length)
        batch_tokens2 = model_tokens[j]
        batch_mean_entropy1, batch_mean_entropy2 = model_mean_entropy[i], model_mean_entropy[j]
        batch_token_probs1, batch_token_probs2 = model_token_probs[i], model_token_probs[j]
        assert len(batch_tokens1) == len(batch_tokens2), "length error!"
        
        edit_distacne, jaccard_similarity, entropy_simlarity  = [], [], []
        # compute mean jaccard similarity
        for k in range(len(batch_tokens1)):
            tokens1, tokens2 = batch_tokens1[k], batch_tokens2[k]
            mean_entropy1, mean_entropy2 = batch_mean_entropy1[k], batch_mean_entropy2[k]
            token_probs1, token_probs2 = batch_token_probs1[k], batch_token_probs2[k]
            assert len(tokens1) == len(token_probs1), "length error!"
            
            edit_distacne.append(weighted_edit_distance(tokens1, tokens2, token_probs1, token_probs2))
            jaccard_similarity.append(jaccard_similarity(tokens1, tokens2))
            entropy_simlarity.append(abs(mean_entropy1 - mean_entropy2))
        
        # Compute the mean value of each fingerprint.
        l = len(batch_tokens1)
        v1, v2, v3 = sum(edit_distacne) / l, jaccard_similarity / l, entropy_simlarity / l
        edit_distance_matrix[i][j], jaccard_simlarity_matrix[i][j], entropy_simlarity_matrix[i][j] = v1, v2, v3
        edit_distance_matrix[j][i], jaccard_simlarity_matrix[j][i], entropy_simlarity_matrix[j][i] = v1, v2, v3
        
    return edit_distance_matrix, jaccard_simlarity_matrix, entropy_simlarity_matrix
            

if __name__ == '__main__':
    trigger_set = load_from_disk('./data/final_trigger_set')
    x, y, z = evaluate(trigger_set=trigger_set, model_list=MODEL_LIST)