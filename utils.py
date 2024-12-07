import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from datasets import Dataset, load_from_disk
import random
from model_list import *
from typing import Sequence, Dict
from tqdm import tqdm
from torch.utils.data import Dataset as ds
from dataclasses import dataclass, field

def load_hf_model(model_name_or_path, 
                  generation_mode=False,
                  bnb_config=None, 
                  fine_tuned=False, 
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
        if generation_mode:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True
                                                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                      use_fast=False,
                                                      padding_side='left',
                                                      )
            return model, tokenizer
        
        if "instruction_tuning_models" in model_name_or_path in model_name_or_path:
            fine_tuned = True
        if not fine_tuned:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True
                                                        )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                      use_fast=False,
                                                      padding_side='left',
                                                      )
        # load lora model
        else:
            # torch_dtype=torch.bfloat16, quantization_config=bnb_config
            config = PeftConfig.from_pretrained(model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True,
                                                        # torch_dtype=torch.bfloat16,
                                                        # quantization_config=bnb_config
                                                        )
            model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                        return_dict=True, 
                                                        device_map="auto",
                                                        output_hidden_states=True,
                                                        torch_dtype=torch.bfloat16,
                                                        quantizaiton_config=bnb_config
                                                        )
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def construct_contrastive_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    ):
    """
    Constructs contrastive learning samples for each data point.
    
    Returns:
    - A new dataset with (input_ids, attention_mask) structure.
    """
    
    model_number = len(MODEL_LIST)
    raw_data = load_from_disk('./data/trajectory_set')
    prompt_number = len(raw_data) // model_number
    
    def prob_map_fun(example):
        return str(round(example, 2))
    
    template = "Prompt: {}<SEP>Output: {}<SEP>Mean Entropy: {}.<SEP>Token Probs: {}."
    contrastive_dataset = []
    print(f"Contructing the dataset for contrastive learning...")
    for i in tqdm(range(len(raw_data))):
        prompt = raw_data[i]['prompt']
        output = raw_data[i]['output']
        mean_entropy = str(raw_data[i]['mean_entropy'])
        token_probs = raw_data[i]['token_probs']
        token_probs = "-".join(list(map(prob_map_fun, token_probs)))
        contrastive_dataset.append(template.format(prompt, output, mean_entropy, token_probs))
    print(f"Tokenizing the dataset...")
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
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
    # dataset.save_to_disk('./data/contrastive_set')
    
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