import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from typing import List
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from utils import *
from model_list import *

def calculate_dimension_difference(
    victim_model, 
    suspected_model, 
    tokenizer, 
    query_set, 
    N: int,
    error_term: float
):
    """
    Calculate the dimension difference (Delta r) between a victim model and a suspected model.

    Parameters:
    -----------
    victim_model_name : str
        The name of the victim model (e.g., from Hugging Face's model hub).
    suspected_model : callable
        The suspected model to query logits for.
    query_set : List[str]
        A set of queries used to calculate the difference.
    N : int
        The least number of samples to determine the dimension difference.
    error_term : float
        The error threshold to decide when dimensions differ.

    Returns:
    --------
    Delta_r : int
        The calculated dimension difference.
    """
    # # Load the victim causal language model
    # victim_model = AutoModelForCausalLM.from_pretrained(victim_model_name)
    # victim_model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(victim_model_name)
    
    # Get the weight matrix of the final linear layer
    W = victim_model.lm_head.weight.detach()  # Shape: (vocab_size, hidden_size)
    W = W.cpu().numpy()  # Convert to NumPy for matrix operations

    # Initialize parameters
    delta_r = 0
    n = 0
    S = []  # Store the sampled logits
    
    # query_set = list(query_set['prompt'])
    # Step 1: Query the suspected model
    while n <= N:
        # Randomly sample a query q
        query = np.random.choice(query_set)
        
        # Tokenize the query and get the logits from the suspected model
        inputs = tokenizer(query, return_tensors="pt").to(suspected_model.device)
        with torch.no_grad():
            logits = suspected_model(
                **inputs, 
        ).logits  # Shape: (batch_size, seq_length, vocab_size)
        
        # Aggregate the logits (e.g., take the last token output)
        O = logits.view(-1, logits.size(-1)).cpu().numpy() # (batch_size * seq_length, vocab_size)
        # O = logits[:, -1, :].squeeze().cpu().numpy()  # Shape: (vocab_size,)
        for logit in O:
            S.append(logit)
        # S.append(O)
        n = len(S)
    if n > N:
        S = S[: N]
    
    # Step 2: Calculate the dimension difference
    for i in tqdm(range(len(S))):
        # Solve Wx = s_i to get x_hat
        si = S[i]  # i-th sampled logits
        
        # Handle the dimension issues. 
        ground_truth_dim = len(W[0])
        if len(si) < ground_truth_dim:
            si = np.pad(si, (0, ground_truth_dim - len(si)), mode='constant', constant_values=0)
        else:
            si = si[:ground_truth_dim]
            
        x_hat, _, _, _ = np.linalg.lstsq(W, si, rcond=None)  # Solve W * x_hat ≈ s_i
        
        # Calculate the reconstruction error (di)
        reconstruction = np.dot(W, x_hat)
        di = np.linalg.norm(si - reconstruction)
        
        # If the error exceeds the threshold, increment Delta_r
        if di > error_term:
            delta_r += 1
    
    return delta_r

def run(
    model_list: List[str], 
    victim_family: str, 
    sample_size: int, 
):
    # Fix the random seed. 
    set_seed(42)
    
    # Prompt set. 
    query_set = load_from_disk('./data/seed_trigger_set')
    # Random sampling. 
    query_set = query_set.shuffle(seed=42)
    query_set = query_set.select(range(sample_size))
    # Transform to the array. 
    query_set = list(query_set['prompt'])
    
    assert victim_family in ['llama', 'qwen', 'mistral'], "not a legel family!"
    if victim_family == 'llama':
        victim_model, tokenizer = load_hf_model(
            '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
        )
        index = 0
    elif victim_family == 'qwen':
        victim_model, tokenizer = load_hf_model(
            '/mnt/data/hf_models/qwen-2.5-7b-instruct/Qwen2.5-7B-Instruct', 
        )
        index = 1
    else:
        victim_model, tokenizer = load_hf_model(
            '/mnt/data/hf_models/mistral-7b-instruct/Mistral-7B-Instruct-v0.1', 
        )
        index = 2
        
    model_list = np.array(model_list)
    model_list_ = np.delete(model_list, index)
    
    preds = []
    for model in tqdm(model_list_):
        suspected_model, _ = load_hf_model(
            model_name_or_path=model
        )
        
        delta_r = calculate_dimension_difference(
            victim_model=victim_model, 
            suspected_model=suspected_model, 
            tokenizer=tokenizer, 
            query_set=query_set, 
            N=10, 
            error_term=1e-5, 
        )
        
        preds.append(delta_r)

    # Evaluation.
    model_number = len(model_list)
    labels = np.zeros(model_number)
    labels[index : index + 10] = 1
    labels_ = np.delete(labels, index)
    assert len(labels_) == len(preds), "length error! Line 150."
    
    fpr, tpr, thresholds = roc_curve(y_true=labels_, y_scores=preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    print(f"ROC-AUC Score of {victim_family.capitalize()}: {roc_auc_score(y_true=labels_, y_score=preds)}")
    
    print(preds)
    
if __name__ == "__main__":
    

    # Calculate the dimension difference
    run(
        model_list=MODEL_LIST_TRAIN, 
        victim_family='llama', 
        sample_size=300, 
    )
    # print(f"Dimension Difference (Delta r): {delta_r}")
