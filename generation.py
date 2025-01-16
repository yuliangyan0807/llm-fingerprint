import torch
import torch.nn.functional as F
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
from typing import List
import warnings

def generation(model_name_or_path: str,
               prompt: str,
               temperature: float=1.0,
               fine_tuned=False
               ):
    
    if "instruction_tuning_models" in model_name_or_path in model_name_or_path:
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
        model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                    return_dict=True, 
                                                    device_map="auto",
                                                    output_hidden_states=True
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
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
        "max_new_tokens":128,
        "do_sample":False,
        # "top_k":3,
        # "top_p":0.9,
        # "temperature": temperature,
        # "repetition_penalty":1.4,
        "pad_token_id":tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        # {sequences: , scores: , hidden_states: , attentions: }
        output = model.generate(**generation_input)
    
    print("Current model: {}".format(model_name_or_path))

    gen_sequences = output.sequences[:, input_ids.shape[-1]:]
    decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
    # print("Generated content is: {}".format(decoded_output))
    
    # batch_decoded_output = tokenizer.batch_decode(output.sequences)[0]
    # print(batch_decoded_output)

    tgt_len0 = output.sequences.size()[-1] - input_ids.size()[-1]
    tgt_len = len(output.scores)
    assert tgt_len == tgt_len0

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
    
    tokens = [tokenizer.decode(token_id) for token_id in id_list]
    # print("tokens: {}".format(tokens))
    
    # return id_list, token_probs
    return tokens, token_probs, decoded_output[0]

def batch_generation(
                    model,
                    tokenizer,
                    prompt: List[str],
                    max_new_tokens: int=64,
                    ):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = 0

    # (batch_size, max_length)
    input_ids = tokenizer(
                        prompt, 
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding=True,
                        ).input_ids

    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)

    model.eval()

    generation_input = {
        "input_ids":input_ids,
        "return_dict_in_generate":True,
        "output_scores":True,
        "output_logits":True,
        #"output_hidden_states":True,
        "max_new_tokens":max_new_tokens,
        "do_sample":False,
        # "top_k":3,
        # "top_p":0.9,
        # "temperature": temperature,
        "repetition_penalty":1.4,
        # "pad_token_id":tokenizer.eos_token_id,
        "pad_token_id":0,
    }
    
    with torch.no_grad():
        # {sequences: , scores: , hidden_states: , attentions: }
        output = model.generate(**generation_input)

    gen_sequences = output.sequences[:, input_ids.shape[-1]:] # token_ids: (batch_size, max_gen_length)
    try:
        decoded_output = [tokenizer.decode(ids) for ids in gen_sequences] # texts: (batch_size, text_length))
    except Exception as e:
        decoded_output = ["" for _ in range(len(gen_sequences))]
        pass
    
    # attention mask for filtering the padding tokens
    attention_mask = torch.where(gen_sequences == 0, 0, 1)
    
    # compute the entropy
    logits = torch.stack([logit for logit in output.logits], dim=0).permute(1, 0, 2) # (batch_size, seq_length, vocab_size)
    probs = torch.softmax(logits, dim=-1)
    entropy = torch.sum(-(probs * torch.log2(probs + 1e-12)), dim=-1) * attention_mask
    entropy = [list(filter(lambda x: x != 0, seq)) for seq in entropy]
    entropy = [[t.item() for t in seq] for seq in entropy]
    mean_entropy = [sum(seq) / len(seq) for seq in entropy]
    # varentropy = torch.var(entropy, dim=-1)

    token_probs = [[] for _ in range(len(prompt))]

    # batch loop
    for i in range(len(output.scores)):
        batch_score = output.scores[i] # (batch_size, vocab_size)
        # Convert the scores to probabilities
        probs = torch.softmax(batch_score, -1) # (batch_size, vocab_size)
        # Take the probability for the generated tokens (at position i in sequence)
        # Iterate each batch
        for j in range(len(probs)):
            # filter out the pad token
            if gen_sequences[j, i].item() != 0:
                try:
                    token_probs[j].append(probs[j, gen_sequences[j, i].item()].item())
                except IndexError as e:
                    continue
            else:
                continue
    # some bugs with gemma2       
    batch_tokens = []
    for token_ids in gen_sequences:
        tokens = []
        for token_id in token_ids:
            # filter out the pad token
            if token_id != 0:
                tokens.append(tokenizer.decode(token_id))   
        batch_tokens.append(tokens)
    
    return batch_tokens, token_probs, decoded_output, entropy, mean_entropy

if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')
    
    # debug
    model, tokenizer = load_hf_model("/mnt/data/yuliangyan/meta-llama/Meta-Llama-3-8B/",)
    
    prompts = [
    # "Once upon a time,",
    # "In a galaxy far, far away,",
    # "Artificial intelligence can",
    # "The future of technology",
    "Let G be a group of order 35. What can be said about G?  Answer Choices: (A) G must be abelian. (B) G must be cyclic. (C) G must be a direct product of cyclic groups. (D) G cannot be cyclic."
    ]
    
    tokens, token_probs, decode_output, entropy, mean_entropy = batch_generation(model=model, 
                                                          tokenizer=tokenizer, 
                                                          prompt=prompts,
                                                          max_new_tokens=32)
    print(tokens)
    print(token_probs)
    print(decode_output)
    print(entropy)
    print(mean_entropy)