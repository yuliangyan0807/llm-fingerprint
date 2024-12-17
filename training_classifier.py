import torch
import torch.nn.functional as F
import transformers
from transformers import T5EncoderModel, set_seed
from torch.utils.data import random_split
from dataclasses import dataclass, field
from typing import Optional
import wandb

from utils import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google-t5/t5-base")
    num_labels: int = field(
        default=9,
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="wandb")
    run_name: str = field(default='llm-fingerprint-1109')
    max_grad_norm: str = field(default=1.0)
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

class ContrastiveTrainer(transformers.Trainer):
    """
    Customize trainer for contrastive learning.
    """
    
    def compute_loss(self, 
                     model, 
                     inputs,
                     temperature=0.2, 
                     ):
        """
        inputs: Dict{"input_ids", "attention_mask"}.
        """
        total_loss = 0.0
        count = 0
        batch_input_ids = inputs['input_ids'] # (batch_size, 18, seq_length)
        batch_attention_mask = inputs['attention_mask']
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            count += 1
            # input_ids = sample['input_ids']
            # attention_mask = sample['attention_mask']
            model_intpus = {
                "input_ids" : torch.tensor(input_ids),
                "attention_mask" : torch.tensor(attention_mask),
            }   
            # print(model_intpus['input_ids'].shape)
            hidden_states = model(**model_intpus).last_hidden_state # (18, seq_length, hidden_dim)
            # Aggeragate the hidden states.
            aggeragated_hidden_states = torch.sum(hidden_states, dim=1) # (18, hidden_dim)
            # aggeragated_hidden_states = hidden_states[:, 0, :] # (18, hidden_dim)
            aggeragated_hidden_states = F.normalize(aggeragated_hidden_states, p=2, dim=-1)
            similarity_matrix = torch.matmul(aggeragated_hidden_states, aggeragated_hidden_states.T) / temperature # (18, 18)
                
            num_samples = similarity_matrix.size(0)
            positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
            for i in range(0, num_samples, 2):
                positive_mask[i][i + 1] = 1
                positive_mask[i + 1][i] = 1
                
            # # Get positive pair (anchor, positive sample) similarities,
            # positive_logits = similarity_matrix[positive_mask]

            # # Mask out (anchor, positive sample) and (anchor, anchor) pairs for the denominator
            # denominator_mask = ~positive_mask & ~torch.eye(num_samples, dtype=torch.bool, device=similarity_matrix.device)
            # negative_logits = torch.exp(similarity_matrix[denominator_mask]).view(num_samples, -1) # (18, 16)

            # # Compute the InfoNCE loss
            # loss = -torch.log(torch.exp(positive_logits) / negative_logits.sum(dim=1)).mean()
            # # loss = -torch.log(torch.exp(positive_logits)).sum()
            # total_loss += loss
            
            # Mask out diagonal (self-similarity)
            diagonal_mask = torch.eye(num_samples, dtype=torch.bool).to(model.device)
            # similarity_matrix.masked_fill_(diagonal_mask, float('inf'))
            # Notice that this may casue the overflow.
            similarity_matrix.masked_fill_(diagonal_mask, -1e9)
            
            # Apply log-softmax across rows
            logits = F.log_softmax(similarity_matrix, dim=1)
            
            # Only select positive logits
            positive_logits = logits[positive_mask]
            
            # Calculate the contrastive loss for positive pairs
            # loss = -positive_logits.mean()
            loss = -positive_logits.sum()
            total_loss += loss

        return total_loss / count

def train():
    
    set_seed(42)

    os.environ["WANDB_PROJECT"]="llm-fingerprint"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = T5EncoderModel.from_pretrained(model_args.model_name_or_path,
                                                 output_hidden_states=True
                                                # num_labels=model_args.num_labels,
                                                )
    # Some bugs about the T5 model.
    model.floating_point_ops = lambda s: 0
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    contrastive_dataset = ContrastiveDataset(construct_contrastive_dataset(tokenizer=tokenizer))
    data_collator = DataCollatorForContrastiveDataset(tokenizer=tokenizer)
    
    train_size = int(0.9 * len(contrastive_dataset))
    test_size = len(contrastive_dataset) - train_size
    train_dataset, test_dataset = random_split(contrastive_dataset, [train_size, test_size])
    
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    wandb.finish()

if __name__ == '__main__':

    train()