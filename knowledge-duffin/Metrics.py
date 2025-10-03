from Utils import *
import json
import re
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from collections import defaultdict

def extract_answer(text):
    """Extracts the predicted answer from generated text."""
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return extract_again(text)

def extract_again(text):
    """Fallback extraction method."""
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    return match.group(1) if match else extract_final(text)

def extract_final(text):
    """Final fallback extraction method."""
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None


def HiMing_similarity(tokens1, tokens2):
    """Computes similarity between two token lists based on identical elements."""
    if len(tokens1) != len(tokens2):
        raise ValueError("Both lists must have the same length")
    return sum(1 for a, b in zip(tokens1, tokens2) if a == b) / len(tokens1)

def evaluate_AUC_ROC_folds():
    """Evaluates AUC-ROC performance across multiple folds."""
    all_similarity_dict = defaultdict(lambda: defaultdict(float))
    basemodels = ["Meta-Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct", "Llama-3.2-3B-Instruct"]
    y_trues, y_scores = [], []
    
    # Group models into categories
    llama = [model.split('/')[-1] for model in list(Llama.keys())]
    qwen = [model.split('/')[-1] for model in list(Qwen.keys())]
    mistral = [model.split('/')[-1] for model in list(Mistral.keys())]
    
    for basemodel in basemodels:
        postiveset = llama if basemodel == basemodels[0] else qwen if basemodel == basemodels[1] else mistral
        
        # Simulating multiple runs (batches of test cases)
        for i in range(1):
            extracted_answers = extract_answers("./output", 20)
            extended_indices = {m: [idx for cat, idx in categories.items()] for m, categories in extracted_answers.items()}
            
            similarity_results, y_true, y_score = {}, [], []
            basemodelset = extended_indices.get(basemodel, [])
            
            for modelname, model_indices in extended_indices.items():
                if modelname == basemodel:
                    all_similarity_dict[basemodel][basemodel] += 1
                else:
                    sim_value = HiMing_similarity(basemodelset, model_indices)
                    similarity_results[modelname] = sim_value
                    all_similarity_dict[basemodel][modelname] += sim_value
                    y_true.append(1 if modelname in postiveset else 0)
                    y_score.append(sim_value)
            
            y_trues.append(y_true)
            y_scores.append(y_score)
    
    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
    y_trues, y_scores = y_trues.reshape(4, 5, 32), y_scores.reshape(4, 5, 32)
    y_trues, y_scores = y_trues.transpose(1, 0, 2), y_scores.transpose(1, 0, 2)
    
    output_file = "roc_data.json"
    with open(output_file, "w") as f:
        json.dump({"y_true": y_trues.tolist(), "y_score": y_scores.tolist()}, f)
    print(f"ROC data saved to {output_file}")
