import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def calculate_dimension_difference(
    victim_model_name: str,
    suspected_model,
    query_set: List[str],
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
    # Load the victim causal language model
    victim_model = AutoModelForCausalLM.from_pretrained(victim_model_name)
    victim_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(victim_model_name)
    
    # Get the weight matrix of the final linear layer
    W = victim_model.lm_head.weight.detach()  # Shape: (vocab_size, hidden_size)
    W = W.cpu().numpy()  # Convert to NumPy for matrix operations

    # Initialize parameters
    Delta_r = 0
    n = 0
    S = []  # Store the sampled logits
    
    # Step 1: Query the suspected model
    while n <= N:
        # Randomly sample a query q
        query = np.random.choice(query_set)
        
        # Tokenize the query and get the logits from the suspected model
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            logits = suspected_model(**inputs).logits  # Shape: (batch_size, seq_length, vocab_size)
        
        # Aggregate the logits (e.g., take the last token output)
        O = logits[:, -1, :].squeeze().cpu().numpy()  # Shape: (vocab_size,)
        S.append(O)
        n = len(S)
    
    # Step 2: Calculate the dimension difference
    for i in range(n):
        # Solve Wx = s_i to get x_hat
        si = S[i]  # i-th sampled logits
        x_hat, _, _, _ = np.linalg.lstsq(W, si, rcond=None)  # Solve W * x_hat ≈ s_i
        
        # Calculate the reconstruction error (di)
        reconstruction = np.dot(W, x_hat)
        di = np.linalg.norm(si - reconstruction)
        
        # If the error exceeds the threshold, increment Delta_r
        if di > error_term:
            Delta_r += 1
    
    return Delta_r

if __name__ == "__main__":
    # Victim and suspected model configuration
    victim_model_name = "gpt2"
    
    # Assuming the suspected model behaves like the victim model for demonstration
    suspected_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Define the query set and parameters
    query_set = ["Hello world!", "How are you?", "This is a test.", "Causal language models are powerful."]
    N = 10
    error_term = 1e-3

    # Calculate the dimension difference
    delta_r = calculate_dimension_difference(victim_model_name, suspected_model, query_set, N, error_term)
    print(f"Dimension Difference (Delta r): {delta_r}")
