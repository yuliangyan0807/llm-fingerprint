import torch
import numpy as np
import itertools
from tqdm import tqdm
from typing import Union, Dict, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

from test_prompt import *
from metrics import *
from model_list import *


def extract_fingerprint(model_name_or_path: str,
                        prompt: str,
                        fine_tuned=False):
    if "test" in model_name_or_path:
        fine_tuned = True
    if not fine_tuned:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # load lora model
    else:
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        peft_model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # input_ids  size[1,12]
    # prompt = "please prove that: $\\lim_{n\to\\infty} \\sin{n}$ does not exist."
    # prompt = "Teach children how to access and use firearms?"
    input_ids = tokenizer(prompt, 
                        return_tensors="pt",
                        add_special_tokens=False
                        ).input_ids
    #tokenized_chat = tokenizer.apply_chat_template(input_text, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)

    model.eval()

    generation_input = {
        "input_ids":input_ids,
        "return_dict_in_generate":True,
        "output_scores":True,
        #"output_hidden_states":True,
        "max_new_tokens":256,
        "do_sample":True,
        # "top_k":50,
        "top_p":0.9,
        "temperature":0.45,
        "repetition_penalty":1.4,
        # "eos_token_id":tokenizer.eos_token_id,
        # "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.eos_token_id,

    }

    with torch.no_grad():
        # {sequences: , scores: , hidden_states: , attentions: }
        output = model.generate(**generation_input)
    
    print("Current model: {}".format(model_name_or_path))

    gen_sequences = output.sequences[:, input_ids.shape[-1]:]
    decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
    print("Generated content is: {}".format(decoded_output))
    print("###################################################")
    
    # batch_decoded_output = tokenizer.batch_decode(output.sequences)[0]
    # print(batch_decoded_output)

    tgt_len0 = output.sequences.size()[-1] - input_ids.size()[-1]
    tgt_len = len(output.scores)
    assert tgt_len == tgt_len0
    print("Generated length is {}.".format(tgt_len0))
    print("###################################################")

    token_probs = []
    index = []

    # TODO
    # batch loop
    for i, score in enumerate(output.scores):
        # Convert the scores to probabilities
        probs = torch.softmax(score, -1)
        # Take the probability for the generated tokens (at position i in sequence)
        token_probs.append(probs[0, gen_sequences[0, i]].item())
        index.append(i)

    id_list = gen_sequences.tolist()[0]
    
    # for token_id, prob in zip(id_list, token_probs):
    #     print(f"{tokenizer.decode(token_id)}: {prob}")
    
    tokens = [tokenizer.decode(token_id) for token_id in id_list]
    print("tokens: {}".format(tokens))
    
    # return id_list, token_probs
    return tokens, token_probs
    

def get_distance_matrix(seed_prompt: str):
    
    num_models = len(MODEL_LIST)
    distance_matrix = np.zeros((num_models, num_models))
    
    fingerprint = []
    for i in range(num_models):
        id_list, token_probs = extract_fingerprint(MODEL_LIST[i], seed_prompt)
        fingerprint.append((id_list, token_probs))
        # print(MODEL_LIST[i])
    
    for i, j in itertools.combinations(range(num_models), 2):
        
        # model_name_or_path_0 = MODEL_LIST[i]
        # model_name_or_path_1 = MODEL_LIST[j]
        
        id_list_0, token_probs_0 = fingerprint[i][0], fingerprint[i][1]
        id_list_1, token_probs_1 = fingerprint[j][0], fingerprint[j][1]

        # distance_matrix[i, j] = weighted_edit_distance(id_list_0,
        #                                                id_list_1,
        #                                                token_probs_0,
        #                                                token_probs_1
        #                                                )
        distance_matrix[i, j] = edit_distance(id_list_0,
                                              id_list_1)
        distance_matrix[j, i] = distance_matrix[i, j]
        
    return distance_matrix

def trace_plot(token_probs,
               model_name_or_path: str
               ):
    # assert len(index) == len(token_probs)
    # index = np.array(index)
    index = np.arange(0, len(token_probs))
    token_probs = np.array(token_probs)
    
    fig_name = model_name_or_path.rstrip('/')
    fig_name = fig_name.split('/')[-1]
    
    plt.plot(index, token_probs, color='red')
    plt.title('Inference Trajectory of {}'.format(fig_name))
    plt.xlabel('Tokens')
    plt.ylabel("Probability")
    plt.xlim(0, 127)
    plt.ylim(0, 1)
    
    plt.grid(True)
    
    save_name = fig_name + '.png'
    plt.savefig(save_name, dpi=300)
    plt.show()

def distance_matrix_plot(distance_matrix,
                         save_dir
                         ):
    
    num_models = len(MODEL_LIST)
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=MODEL_LIST, yticklabels=MODEL_LIST)

    plt.title("Distance Matrix Heatmap", fontsize=16)
    plt.xlabel("Model Index", fontsize=12)
    plt.ylabel("Model Index", fontsize=12)

    plt.savefig(save_dir, format='png', dpi=300)

    plt.show()

if __name__ == '__main__':

    # model_name_or_path = "/mnt/data/yuliangyan/bigscience/bloom-7b1"
    # model_name_or_path_0 = "/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/"
    # model_name_or_path_1 = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_8_test"

    prompt = EXAMPLE_5
    
    # id_list_0, token_probs_0 = extract_fingerprint(model_name_or_path_0, prompt)
    # id_list_1, token_probs_1 = extract_fingerprint(model_name_or_path_1, prompt, fine_tuned=True)
    
    # dis = weighted_edit_distance(id_list_0, id_list_1, token_probs_0, token_probs_1)
    # print(dis)
    
    
    distance_matrix = get_distance_matrix(seed_prompt=prompt)
    print(distance_matrix)
    distance_matrix_plot(distance_matrix, "test_5_w_o.png")
    
    # trace_plot(token_probs, model_name_or_path)