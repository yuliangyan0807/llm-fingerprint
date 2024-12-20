from metrics import *
from generation import *
from typing import List
from utils import load_hf_model
from itertools import combinations
import numpy as np
from tqdm import tqdm
from model_list import *
from datasets import load_from_disk
# from fig_plot import *
from transformers import T5EncoderModel, set_seed
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from sklearn.metrics import roc_curve, auc, roc_auc_score

@torch.no_grad()
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
            jaccard_similarity.append(compute_jaccard_similarity(tokens1, tokens2))
            entropy_simlarity.append(abs(mean_entropy1 - mean_entropy2))
        
        # Compute the mean value of each fingerprint.
        l = len(batch_tokens1)
        v1, v2, v3 = sum(edit_distacne) / l, sum(jaccard_similarity) / l, sum(entropy_simlarity) / l
        edit_distance_matrix[i][j], jaccard_simlarity_matrix[i][j], entropy_simlarity_matrix[i][j] = v1, v2, v3
        edit_distance_matrix[j][i], jaccard_simlarity_matrix[j][i], entropy_simlarity_matrix[j][i] = v1, v2, v3
    
    res = {'edit_distance_matrix' : edit_distance_matrix,
           'jaccard_simlarity_matrix' : jaccard_simlarity_matrix,
           'entropy_simlarity_matrix' : entropy_simlarity_matrix,
           }
    
    return res

@torch.no_grad()
def evaluate_cl(
    model_list,
    contrastive_set,
    model_name_or_path: str,
    # batch_size: int,
):  
    model = T5EncoderModel.from_pretrained(
        model_name_or_path,
        # device_map='auto',
        output_hidden_states=True,
        ignore_mismatched_sizes=True,
        )
    
    # l = len(test_trigger_set)
    simlarity_marices = [] # (len(test_trigger_set), 18, 18)
    
    for sample in tqdm(contrastive_set):
        batch_input_ids = sample['input_ids'] # (18, seq_length)
        batch_attention_mask = sample['attention_mask']
        inputs = {
            'input_ids' : torch.tensor(batch_input_ids),
            'attention_mask' : torch.tensor(batch_attention_mask),
        }
        last_hidden_states = model(**inputs).last_hidden_state # (18, seq_length, hidden_states)
        aggregated_hidden_states = torch.sum(last_hidden_states, dim=-1) # (18, hidden_states)
        aggregated_hidden_states = F.normalize(aggregated_hidden_states, dim=-1)
        simlarity_marix = torch.matmul(aggregated_hidden_states, aggregated_hidden_states.T) # (18, 18)
        simlarity_marices.append(simlarity_marix)
    
    simlarity_marices = torch.stack(simlarity_marices, dim=0)
    # print(simlarity_marices.shape)   

    mean_simlarity_matrix = torch.mean(simlarity_marices, dim=0)
    # print(mean_simlarity_matrix.shape)
    
    print(f"Start Evaluating...")
    
    # Compute the model numbers in each family.
    model_number_per_family = len(model_list) // 3
    model_number = len(model_list)
    original_model_index = range(0, model_number, model_number_per_family)
    
    # Llama Family.
    llama_labels = np.zeros(model_number)
    start = original_model_index[0]
    llama_labels[start: start + model_number_per_family] = 1
    print(f"Llama family labels: {llama_labels}")
    
    for i in range(start, start + model_number_per_family):
        # Obtain the i-th (suspect model) row of the simlarity matrix
        llama_pred = mean_simlarity_matrix.detach().numpy()[i,:]
        llama_roc = roc_auc_score(y_true=llama_labels, y_score=llama_pred)
        current_model = model_list[i][model_list[i].rfind('/') + 1 : ]
        print(f"Suspect Model is {current_model}.")
        print(f"Extractor predict on current: {llama_pred}")
        print(f"Roc-auc score of the current: {llama_roc}")
        print(f"#############################################")
    
    # Qwen Family
    qwen_labels = np.zeros(model_number)
    start = original_model_index[1]
    qwen_labels[start: start + model_number_per_family] = 1
    print(f"Qwen family labels: {qwen_labels}")
    
    for i in range(start, start + model_number_per_family):
        qwen_pred = mean_simlarity_matrix.detach().numpy()[i,:]
        qwen_roc = roc_auc_score(y_true=qwen_labels, y_score=qwen_pred)
        current_model = model_list[i][model_list[i].rfind('/') + 1 : ]
        print(f"Suspect Model is {current_model}.")
        print(f"Extractor predict on current: {qwen_pred}")
        print(f"Roc-auc score of the current: {qwen_roc}")
        print(f"#############################################")
    
    # Mistral Family.
    mistral_labels = np.zeros(model_number)
    start = original_model_index[2]
    mistral_labels[start: start + model_number_per_family] = 1
    print(f"Mistral family labels: {mistral_labels}")
    
    for i in range(start, start + model_number_per_family):
        mistral_pred = mean_simlarity_matrix.detach().numpy()[i,:]
        mistral_roc = roc_auc_score(y_true=mistral_labels, y_score=mistral_pred)
        current_model = model_list[i][model_list[i].rfind('/') + 1 : ]
        print(f"Suspect Model is {current_model}.")
        print(f"Extractor predict on current: {mistral_pred}")
        print(f"Roc-auc score of the current: {mistral_roc}")
        print(f"#############################################")
    
    
if __name__ == '__main__':
    set_seed(42)
        
    # evaluate trajectories
    # trigger_set = load_from_disk('./data/final_trigger_set')
    
    # res = evaluate(trigger_set=trigger_set, model_list=MODEL_LIST)
    
    # distance_matrix_plot(result_dict=res,
    #                      save_dir='result.png')
    
    
    # evaluate cl classifier
    # model_path = './metric_learning_models/1220_0'
    model_path = "google-t5/t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # evaluation on the train models.
    raw_data = load_from_disk('./data/trajectory_set_train')
    contrastive_dataset_evaluation = ContrastiveDataset(construct_contrastive_dataset(tokenizer=tokenizer,
                                                                                      raw_data=raw_data,
                                                                                      model_list=MODEL_LIST_TRAIN))
    

    # evaluation on the test models.
    # raw_data = load_from_disk('./data/trajectory_set_evaluation')
    # contrastive_dataset_evaluation = ContrastiveDataset(construct_contrastive_dataset(tokenizer=tokenizer,
    #                                                                                   raw_data=raw_data,
    #                                                                                   model_list=MODEL_LIST_TEST))
    
    
    evaluate_cl(model_list=MODEL_LIST_TRAIN,
                contrastive_set=contrastive_dataset_evaluation,
                model_name_or_path=model_path,
            )