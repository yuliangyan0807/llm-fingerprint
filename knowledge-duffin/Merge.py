from sklearn.metrics import roc_auc_score

from model_list import get_suspect_models, get_unseen_models
from Utils import load_similarity_data

def merge_unseen():
    """Merges similarities to evaluate unseen models in different cases."""
    knowledge_similarity, base_similarity = load_similarity_data()
    knowledge_similarity.update(base_similarity)

    auc_rocs = {}
    y_true_base = [1, 1, 0, 0, 0]
    y_true_suspect = [1, 0, 0, 0]
    a, b = 0.95, 0.8

    unseen_models = get_unseen_models()
    
    for i in range(3):
        for j, model in enumerate(unseen_models[:3]):
            y_true = y_true_base if model == "Llama-3.2-3B-Instruct" else y_true_suspect
            pred_list = unseen_models[1:] if model == "Llama-3.2-3B-Instruct" else [unseen_models[0]] + unseen_models[3:]
            knowledge_val = [knowledge_similarity[model][m] for m in pred_list]
            y_pred = [a * k + b * 0 for k in knowledge_val]  # Prediction values are masked
            
            if model not in auc_rocs:
                auc_rocs[model] = []
            auc_rocs[model].append(roc_auc_score(y_true, y_pred))
    
    return auc_rocs

def merge():
    """Merges similarities for model evaluation."""
    knowledge_similarity, base_similarity = load_similarity_data()
    knowledge_similarity.update(base_similarity)
    suspect_models = get_suspect_models()

    a, b = 0.95, 0.8
    roc_aucs = {}

    for i in range(3):
        for model in suspect_models[i]:
            y_true = [1] + [0] * (len(suspect_models[i]) - 1)
            y_pred = [a * knowledge_similarity[model][m] + b * 0 for m in suspect_models[i] if m != model]  # Masked predictions
            
            roc_aucs[model] = roc_auc_score(y_true, y_pred)
    
    return roc_aucs

if __name__ == "__main__":
    result = merge()
    unseen_result = merge_unseen()
    print("ROC AUC Results:", result)
    print("Unseen Model ROC AUC:", unseen_result)