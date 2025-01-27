import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from datasets import Dataset, load_from_disk
from typing import Sequence, Dict, List
from tqdm import tqdm
from torch.utils.data import Dataset as ds
from dataclasses import dataclass

from model_list import *


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

def construct_contrastive_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    raw_data,
    model_list,
    select_trigger: bool=False, 
    save: bool=False, 
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
    # template = "Prompt: {}<SEP>Output: {}<SEP>Mean Entropy: {}."
    template = "Output: {} <SEP> Mean Entropy: {}."
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
    assert prompt_number == 600, "error!"
    
    # 3-fold cross-validation
    if fold == 0:
        trajectory_set_train = trajectory_set.select(
            list(range(0, 600)) +
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
    # data = load_from_disk('./data/trajectory_set')
    # print(data[360])
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    construct_contrastive_dataset(tokenizer=tokenizer)