import torch
import transformers
from transformers import AutoModelForSequenceClassification, AdamW
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
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


# Tokenize the dataset
def tokenize_function(batch, tokenizer: transformers.PreTrainedTokenizer):
    # Tokenize anchor and positive, then tokenize each negative sample individually
    anchor_encodings = tokenizer(batch['anchor'], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    positive_encodings = tokenizer(batch['positive'], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    negative_encodings = [tokenizer(neg, padding="max_length", truncation=True, max_length=128, return_tensors="pt") for neg in batch['negatives']]

    return {
        'anchor_input_ids': anchor_encodings['input_ids'],
        'anchor_attention_mask': anchor_encodings['attention_mask'],
        'positive_input_ids': positive_encodings['input_ids'],
        'positive_attention_mask': positive_encodings['attention_mask'],
        'negative_input_ids': [ne['input_ids'] for ne in negative_encodings],
        'negative_attention_mask': [ne['attention_mask'] for ne in negative_encodings],
    }

class ContrastiveDataCollator:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        anchor_batch = [{'input_ids': item['anchor_input_ids'], 'attention_mask': item['anchor_attention_mask']} for item in batch]
        positive_batch = [{'input_ids': item['positive_input_ids'], 'attention_mask': item['positive_attention_mask']} for item in batch]
        
        # Collate negative samples separately
        negative_batches = []
        for i in range(len(batch[0]['negative_input_ids'])):
            negative_batch = [{'input_ids': item['negative_input_ids'][i], 'attention_mask': item['negative_attention_mask'][i]} for item in batch]
            negative_batches.append(self.data_collator(negative_batch))

        # Collate anchor and positive samples
        anchor = self.data_collator(anchor_batch)
        positive = self.data_collator(positive_batch)

        return {
            'anchor': anchor,
            'positive': positive,
            'negatives': negative_batches  # List of collated negative batches
        }

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # data = load_from_disk('./data/trajectory_set')
    data = load_from_disk(data_args.data_path)
    contrastive_dataset = create_contrastive_samples(data=data)
    
    # Apply tokenization to the dataset
    tokenized_dataset = contrastive_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.num_labels,
                                                               )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              )
    
    # Instantiate the data collator
    data_collator = ContrastiveDataCollator(tokenizer)
    
    # Create DataLoaders for each split
    train_loader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_dataset['validation'], batch_size=16, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=False, collate_fn=data_collator)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

if __name__ == '__main__':

    dataset = load_from_disk('./data/contrastive_set')
    dataset = dataset.select(range(0, 4))
    # print(len(dataset))
    # print(dataset[10])
    
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    tokenized_dataset = construct_contrastive_dataset(tokenizer=tokenizer)
    print(len(tokenized_dataset[2]['input_ids']))
    
    # train()