import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AdamW, set_seed
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from utils import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google-t5/t5-base")
    num_labes: int = field(
        default=9,
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        # default="paged_adamw_32bit"
        default='adamw_torch'
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_rank: int = field(
        default=8
    )
    lora_alpha: int = field(
        default=32
    )

class ContrastiveTrainer(transformers.Trainer):
    
    def compute_loss(self, 
                     model, 
                     inputs,
                     temperature, 
                    #  return_outputs=False,
                     ):
        """
        inputs: (batch_size, 18, seq).
        """
        total_loss = 0.0
        count = 0
        for sample in inputs:
            count += 1
            # sample: (18, seq_length)
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            model_intpus = {
                "input_ids" : torch.tensor(input_ids).to(model.device),
                "attention_mask" : torch.tensor(attention_mask).to(model.device),
            }
            hidden_states = model(**model_intpus).encoder_last_hidden_state # (18, seq_length, hidden_dim)
            # Aggeragate the hidden states.
            aggeragated_hidden_states = torch.sum(hidden_states, dim=1) # (18, hidden_dim)
            # aggeragated_hidden_states = hidden_states[:, 0, :] # (18, hidden_dim)
            aggeragated_hidden_states = F.normalize(aggeragated_hidden_states, p=2, dim=-1).to(model.device)
            similarity_matrix = torch.matmul(aggeragated_hidden_states, aggeragated_hidden_states.T) / temperature # (18, 18)
            model_number = similarity_matrix.size(0)
            positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            for i in range(0, model_number, 2):
                positive_mask[i][i + 1] = 1
                positive_mask[i + 1][i] = 1
            
            # Mask out diagonal (self-similarity)
            diagonal_mask = torch.eye(model_number, dtype=torch.bool).to(model.device)
            similarity_matrix.masked_fill_(diagonal_mask, float('inf'))
            
            # Apply log-softmax across rows
            logits = F.log_softmax(similarity_matrix, dim=1)
            
            # Only select positive logits
            positive_logits = logits[positive_mask]
            
            # Calculate the contrastive loss for positive pairs
            loss = -positive_logits.mean()
            total_loss += loss
        
        return total_loss / count

def train():
    
    set_seed(42)
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.num_labels,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              )
    
    contrastive_dataset = ContrastiveDataset(construct_contrastive_dataset(tokenizer=tokenizer))
    data_collator = DataCollatorForContrastiveDataset(tokenizer=tokenizer)
    
    train_size = int(0.9 * len(contrastive_dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(contrastive_dataset, [train_size, test_size])
    
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    
    trainer.train()
    
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    tokenized_dataset = construct_contrastive_dataset(tokenizer=tokenizer)
    print(len(tokenized_dataset))
    
    dataset = ContrastiveDataset(raw_dataset=tokenized_dataset)
    data_collator = DataCollatorForContrastiveDataset(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained("google-t5/t5-base",
                                                               output_hidden_states=True)
    # print(model)
    input_ids = tokenized_dataset[0]['input_ids']
    attention_mask = tokenized_dataset[0]['attention_mask']
    input = {
        'input_ids' : torch.tensor(input_ids),
        'attention_mask' : torch.tensor(attention_mask)
    }
    print(input['input_ids'].shape) # (batch_size, 18, seq_length)
    output = model(**input)
    print(output.encoder_last_hidden_state)
    print(attention_mask)
    print(output.encoder_last_hidden_state.shape)
    aggeragated_hidden_states = torch.sum(output.encoder_last_hidden_state, dim=1) # (18, hidden_dim)
    print(aggeragated_hidden_states.shape)
    aggeragated_hidden_states = F.normalize(aggeragated_hidden_states, p=2, dim=-1)
    similarity_matrix = torch.matmul(aggeragated_hidden_states, aggeragated_hidden_states.T)
    print(similarity_matrix)
    print(similarity_matrix.shape[0])
    
    # print(output.last_hidden_states)
    train()