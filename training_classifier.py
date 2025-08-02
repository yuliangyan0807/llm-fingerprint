import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import T5EncoderModel, set_seed
from dataclasses import dataclass, field
from typing import Optional
import wandb

from utils import *
from model_list import *
from evaluation import *

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
    fold: int = field(default=0)
    seed: int = field(default=42)
    sample_size: int = field(default=0)
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
                     temperature=0.04, 
                     ):
        """
        inputs: Dict{"input_ids", "attention_mask"}.
        """
        batch_input_ids = inputs['input_ids'] # (batch_size, 18, seq_length)
        batch_attention_mask = inputs['attention_mask']
        
        accumulated_similarity_matrix = []
        accumulated_labels = []
        
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            # input_ids = sample['input_ids']
            # attention_mask = sample['attention_mask']
            model_intpus = {
                "input_ids" : torch.tensor(input_ids),
                "attention_mask" : torch.tensor(attention_mask),
            }   
            # print(model_intpus['input_ids'].shape)
            hidden_states = model(**model_intpus).last_hidden_state # (21, seq_length, hidden_dim)
            # TODO
            # Use the representation of the [CLS] token.
            # aggeragated_hidden_states = torch.squeeze(hidden_states[:, 0, :])
            # Aggeragate the hidden states.
            aggeragated_hidden_states = torch.sum(hidden_states, dim=1)
            aggeragated_hidden_states = F.normalize(aggeragated_hidden_states, p=2, dim=-1) # (21, hidden_dim)
            similarity_matrix = torch.matmul(aggeragated_hidden_states, aggeragated_hidden_states.T) / temperature # (21, 21)
            
            # Caculate the total model numbers.
            model_number = similarity_matrix.size(0) # 21
            
            # Select the postive and negative logits.
            remove_indices = torch.arange(0, model_number, step=model_number // 3) # [0, 7, 14]
            all_indices = torch.arange(model_number)
            keep_indices = all_indices[~torch.isin(all_indices, remove_indices)].to(similarity_matrix.device)
            # Remove the raw of the base models.
            similarity_matrix = torch.index_select(similarity_matrix, dim=0, index=keep_indices) # (18, 21)
            # Obtain the logits.
            # (18, 13)
            # (suspect_model, base_model): 1, (suspect_model, other family models): 12.
            logits = torch.cat([
                torch.cat((similarity_matrix[:6, :1], similarity_matrix[:6, 7:]), dim=1),
                torch.cat((similarity_matrix[6:12, :7], similarity_matrix[6:12, 7:8], similarity_matrix[6:12, 14:]), dim=1),
                torch.cat((similarity_matrix[12:, :14], similarity_matrix[12:, 14:15]), dim=1),
            ], dim=0)
            
            # positive labels: [0, 7, 14]
            labels = torch.arange(0, logits.size(0)) // 6 * 7 # (18)
            labels = labels.to(similarity_matrix.device)
            
            accumulated_similarity_matrix.append(logits)
            accumulated_labels.append(labels)
            
        inputs = torch.cat(accumulated_similarity_matrix, dim=0)
        labels = torch.cat(accumulated_labels, dim=0)
        
        loss = nn.CrossEntropyLoss()

        return loss(inputs, labels)

def train():
    
    # set_seed(42)

    os.environ["WANDB_PROJECT"]="llm-fingerprint"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    model = T5EncoderModel.from_pretrained(model_args.model_name_or_path, output_hidden_states=True)
    # Some bugs about the T5 model.
    model.floating_point_ops = lambda s: 0
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # raw_data = load_from_disk('./data/trajectory_set')
    raw_data = load_from_disk(data_args.data_path)
    
    
    fold = training_args.fold
    res, model_list_train, model_list_eval = get_cross_validation_datasets(
        trajectory_set=raw_data,
        model_list=MODEL_LIST_TRAIN,
        fold=fold
    )
    train_set, eval_set = res['train'], res['eval']
    
    contrastive_dataset = ContrastiveDataset(construct_contrastive_dataset(tokenizer=tokenizer,
                                                                           raw_data=train_set,
                                                                           model_list=model_list_train,
                                                                           sample_size=training_args.sample_size))
    data_collator = DataCollatorForContrastiveDataset(tokenizer=tokenizer)
    
    train_dataset = contrastive_dataset
    
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    training_args.output_dir = training_args.output_dir + f"_fold_{fold}"
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluation
    contrastive_dataset = ContrastiveDataset(
        construct_contrastive_dataset(
            tokenizer=tokenizer,
            raw_data=eval_set,
            model_list=model_list_eval,
            # sample_size=training_args.sample_size
        )
    )
    evaluate_cl(
        model_list=model_list_eval,
        contrastive_set=contrastive_dataset,
        model_name_or_path=training_args.output_dir,
    )
    
    wandb.finish()

if __name__ == '__main__':

    train()