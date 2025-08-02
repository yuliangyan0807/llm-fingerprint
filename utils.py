import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from datasets import Dataset, load_from_disk
from typing import Sequence, Dict, List
from tqdm import tqdm
from torch.utils.data import Dataset as ds
from dataclasses import dataclass

from model_list import *
# from generation import batch_generation

def load_hf_model(model_name_or_path, 
                  training_mode=False,
                  device='cuda' if torch.cuda.is_available() else 'cpu'
                  ):
    """
    Load a Hugging Face model, optionally with a LoRA adapter if specified.

    Parameters:
    - model_name_or_path (str): The base model name or path from Hugging Face.
    - fine_tuned (bool): If True, load LoRA weights.
    - device (str): Device to load the model on.

    Returns:
    - model (torch.nn.Module): Loaded model (with LoRA adapter if specified).
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    """
    try:
        # Load base model and tokenizer
        print(f"Loading base model '{model_name_or_path}' on {device}...")
        # load model for generation
        if training_mode:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True
                                                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                      use_fast=False,
                                                      padding_side='right',
                                                      )
            return model, tokenizer
        
        files = [file for file in os.listdir(model_name_or_path)]
        
        # We collected lora models in Llama family.
        lora_models = [
            '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-8bit-Instruct-sql-v3', 
            '/mnt/data/hf_models/llama-3.1-8b-instruct/llama3-8b-sft-qlora-re', 
            '/mnt/data/hf_models/llama-3.1-8b-instruct/prm800k_llama_lora', 
            '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-3_1-8b-instruct-fake-news', 
            '/mnt/data/hf_models/llama-3.1-8b-instruct/task-1-meta-llama-Meta-Llama-3.1-8B-Instruct-1736201342', 
            # '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-2-13b-orca',
        ]
        if model_name_or_path not in lora_models:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True
                                                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                        use_fast=False,
                                                        padding_side='left',
                                                        )
        # Load the Lora fine-tuning version of the model.
        elif model_name_or_path == '/mnt/data/hf_models/llama-3.1-8b-instruct/llama-2-13b-orca':
            
            base_model = AutoModelForCausalLM.from_pretrained(
                '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-2-13b-chat-hf', 
                return_dict=True, 
                device_map='auto', 
                output_hidden_states=True, 
            )
            # model = PeftModel.from_pretrained(
            #     base_model, 
            #     model_name_or_path, 
            # )
            base_model.load_adapter(model_name_or_path)
            
            tokenizer = AutoTokenizer.from_pretrained(
                '/mnt/data/hf_models/llama-3.1-8b-instruct/Llama-2-13b-chat-hf', 
                use_fast=False, 
                padding_side='left', 
            )
            
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
                return_dict=True, 
                device_map='auto', 
                output_hidden_states=True, 
            )
            model = PeftModel.from_pretrained(
                base_model, 
                model_name_or_path, 
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                '/mnt/data/hf_models/llama-3.1-8b-instruct/Meta-Llama-3.1-8B-Instruct', 
                use_fast=False, 
                padding_side='left', 
            )
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded successfully.")
        
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
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

def pre_generate_trajectory(
    model_list: List[str],
    batch_size: int=50,
    max_new_tokens: int=64,
    save_dir: str="",
):
    """
    Given the trigger set, pre generate the trajectory of the models in the model lists.
    Tajectory include tokens, token_probs, entropy, and so on.
    """
    print(f"loading dataset...")
    seed_trigger_set = load_from_disk('./data/seed_trigger_set')
    print(f"{len(seed_trigger_set)} data in total")
    prompts = seed_trigger_set['prompt']
    
    assert len(seed_trigger_set) % batch_size == 0, "choose another batch size!"
    
    tokens_, token_probs_, decode_output_, entropy_, mean_entropy_, labels = [], [], [], [], [], []
    
    # Iterate the model list
    for label, model_name_or_path in enumerate(model_list):
        # load the model
        model, tokenizer = load_hf_model(model_name_or_path)
        # iterate the prompt
        for i in tqdm(range(0, len(seed_trigger_set), batch_size)):
            batch_prompt = prompts[i : i + batch_size]
            tokens, token_probs, decode_output, entropy, mean_entropy = batch_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=batch_prompt,
                max_new_tokens=max_new_tokens
            )
            tokens_ = tokens_ + tokens
            token_probs_ = token_probs_ + token_probs
            decode_output_ = decode_output_ + decode_output
            entropy_ = entropy_ + entropy
            mean_entropy_ = mean_entropy_ + mean_entropy
    
    print(f"saving dataset...")
    optimized_trigger_set = {"tokens": tokens_,
                             "token_probs": token_probs_,
                             "decode_output": decode_output_,
                             "entropy": entropy_,
                             "mean_entropy": mean_entropy_,
                            #  "labels": labels
                             }
    optimized_trigger_set = Dataset.from_dict(optimized_trigger_set)
    # optimized_trigger_set.save_to_disk('./data/trajectory_set_large')
    optimized_trigger_set.save_to_disk(save_dir)

def construct_contrastive_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    raw_data,
    model_list,
    select_trigger: bool=False, 
    save: bool=False, 
    sample_size: int=0,
    include_entropy: bool=True
    ):
    """
    Constructs contrastive learning samples for each data point.
    
    Returns:
    - A new dataset with (input_ids, attention_mask) structure.
    """
    
    # model_number = len(MODEL_LIST_TRAIN)
    model_number = len(model_list)
    # raw_data = load_from_disk('./data/trajectory_set')
    prompt_number = len(raw_data) // model_number
    
    prompts = []
    seed_triggers = load_from_disk('./data/seed_trigger_set')
    for i in range(model_number):
        for j in range(len(seed_triggers)):
            prompt = seed_triggers[j]['prompt']
            prompts.append(prompt)
    
    raw_data = raw_data.add_column('prompt', prompts)
    # raw_data = raw_data.rename_column('')
    print(len(raw_data))
    
    def prob_map_fun(example):
        return str(round(example, 2))
    
    # template = "Prompt: {}<SEP>Output: {}<SEP>Mean Entropy: {}.<SEP>Token Probs: {}."
    if include_entropy:
        template = "Output: {}<SEP>Mean Entropy: {}."
    else:
        template = "Output: {}"
        # template = "Output: {} <SEP> Mean Entropy: {}." 
    contrastive_dataset = []
    print(f"Contructing the dataset for contrastive learning...")
    for i in tqdm(range(len(raw_data))):
        prompt = raw_data[i]['prompt']
        output = raw_data[i]['decode_output']
        mean_entropy = str(raw_data[i]['mean_entropy'])
        token_probs = raw_data[i]['token_probs']
        token_probs = "-".join(list(map(prob_map_fun, token_probs)))
        # contrastive_dataset.append(template.format(prompt, output, mean_entropy, token_probs))
        contrastive_dataset.append(template.format(output, mean_entropy))
    print(f"Tokenizing the dataset...")
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=72,
            truncation=True,
        )
        for text in contrastive_dataset
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # attention_mask = [input_id.ne(tokenizer.pad_token_id) for input_id in input_ids]
    attention_mask = [tokenized.attention_mask for tokenized in tokenized_list]
    assert len(input_ids) == len(raw_data), "length error!"
    
    tokenized_contrastive_dataset= []
    attention_masks = []
    if sample_size != 0:
        prompt_number = sample_size    
    for i in range(prompt_number):
        samples = []
        attn = []
        select_indices = [k * prompt_number + i for k in range(model_number)]
        for j in select_indices:
            samples.append(input_ids[j])
            attn.append(attention_mask[j])
        tokenized_contrastive_dataset.append(samples)
        attention_masks.append(attn)
    
    assert len(tokenized_contrastive_dataset) == prompt_number, "error!"
    data = {'input_ids' : tokenized_contrastive_dataset,
            'attention_mask' : attention_masks,
            }
    dataset = Dataset.from_dict(data)
    if select_trigger: 
        # If we are going to select the triggers, we should not shuffle the dataset. 
        dataset = dataset
    else:
        dataset = dataset.shuffle(seed=42)
    if save == True:
        dataset.save_to_disk('./data/contrastive_set')
    
    return dataset

class ContrastiveDataset(ds):

    def __init__(self, 
                 raw_dataset,
                 ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset

    def __len__(self):
        
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        
        return {
            "input_ids":
            self.raw_dataset[idx]["input_ids"],
            "attention_mask":
            self.raw_dataset[idx]["attention_mask"],
        }

@dataclass
class DataCollatorForContrastiveDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "attention_mask")
        )
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

def get_cross_validation_datasets(
    trajectory_set,
    model_list: List[str],
    fold: int,
):
    res = {}
    prompt_number = len(trajectory_set) // len(model_list)
    # assert prompt_number == 600, "error!"
    
    # 3-fold cross-validation
    if fold == 0:
        trajectory_set_train = trajectory_set.select(
            list(range(0, 1 * prompt_number)) +
            list(range(4 * prompt_number, 10 * prompt_number)) +
            list(range(10 * prompt_number, 11 * prompt_number)) + 
            list(range(14 * prompt_number, 20 * prompt_number)) + 
            list(range(20 * prompt_number, 21 * prompt_number)) + 
            list(range(24 * prompt_number, 30 * prompt_number))
        )
        trajectory_set_eval = trajectory_set.select(
            list(range(0, 4 * prompt_number)) + 
            list(range(10 * prompt_number, 14 * prompt_number)) + 
            list(range(20 * prompt_number, 24 * prompt_number))
        )
        model_list_train = [model_list[i] for i in [0, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29]]
        model_list_eval = [model_list[i] for i in [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]]
        
        res['train'] = trajectory_set_train
        res['eval'] = trajectory_set_eval
        
        return res, model_list_train, model_list_eval
    
    if fold == 1:
        trajectory_set_train = trajectory_set.select(
            list(range(0, 4 * prompt_number)) +
            list(range(7 * prompt_number, 10 * prompt_number)) +
            list(range(10 * prompt_number, 14 * prompt_number)) + 
            list(range(17 * prompt_number, 20 * prompt_number)) + 
            list(range(20 * prompt_number, 24 * prompt_number)) + 
            list(range(27 * prompt_number, 30 * prompt_number))
        )
        trajectory_set_eval = trajectory_set.select(
            list(range(0, 1 * prompt_number)) +
            list(range(4 * prompt_number, 7 * prompt_number)) + 
            list(range(10 * prompt_number, 11 * prompt_number)) +
            list(range(14 * prompt_number, 17 * prompt_number)) + 
            list(range(20 * prompt_number, 21 * prompt_number)) + 
            list(range(24 * prompt_number, 27 * prompt_number))
        )
        model_list_train = [model_list[i] for i in [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29]]
        model_list_eval = [model_list[i] for i in [0, 4, 5, 6, 10, 14, 15, 16, 20, 24, 25, 26]]
        
        res['train'] = trajectory_set_train
        res['eval'] = trajectory_set_eval
        
        return res, model_list_train, model_list_eval
    
    if fold == 2:
        trajectory_set_train = trajectory_set.select(
            list(range(0, 7 * prompt_number)) +
            list(range(10 * prompt_number, 17 * prompt_number)) + 
            list(range(20 * prompt_number, 27 * prompt_number))
        )
        trajectory_set_eval = trajectory_set.select(
            list(range(0, 1 * prompt_number)) +
            list(range(7 * prompt_number, 10 * prompt_number)) + 
            list(range(10 * prompt_number, 11 * prompt_number)) + 
            list(range(17 * prompt_number, 20 * prompt_number)) + 
            list(range(20 * prompt_number, 21 * prompt_number)) + 
            list(range(27 * prompt_number, 30 * prompt_number))
        )
        model_list_train = [model_list[i] for i in [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26]]
        model_list_eval = [model_list[i] for i in [0, 7, 8, 9, 10, 17, 18, 19, 20, 27, 28, 29]]
        
        res['train'] = trajectory_set_train
        res['eval'] = trajectory_set_eval
        
        return res, model_list_train, model_list_eval
         
        
if __name__ == '__main__':
    # Generate the trajectory set.
    pre_generate_trajectory(model_list=MODEL_LIST_TRAIN, save_dir="./data/trajectory_set_0")
    
    # Construct the dataset for training the Trigger-DuFFin.
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    construct_contrastive_dataset(tokenizer=tokenizer)