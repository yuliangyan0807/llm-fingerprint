#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from os.path import exists, join, isdir
import gc
import json
import math
import random
import copy
from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import torch
from torch import nn
from torch.utils.data import Dataset

import transformers
from datasets import load_dataset
from transformers import Trainer, set_seed

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

import warnings

warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


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
    
def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data: {}".format(data_path))
        # data_list = utils.jload(data_path)
        data_list = load_dataset(data_path)['train']
        
        # if select the subset of the raw dataset
        # random_indices = random.sample(range(len(data_list)), 200)
        # data_list = data_list.select(random_indices)

        # Preprocess Data
        logging.warning("Processing data")
        self.tokenizer = tokenizer
        self.sources = []
        self.targets = []

        for idx in range(len(data_list)):
            data = data_list[idx]
            # corpus = data["corpus"]
            corpus = ""
            if corpus != "":
                # pretrain mode
                source = f"{tokenizer.bos_token}"
                self.sources.append(source)

                target = f"{corpus}{tokenizer.eos_token}"
                self.targets.append(target)
            else:
                # instruction mode
                instruction = data["instruction"]
                # conversation = data["conversation"]
                conversation = [instruction]
                if len(conversation) == 1:
                    if instruction == "":
                        source = f"{tokenizer.bos_token}"
                    else:
                        source = f"{tokenizer.bos_token}### System:\n{instruction}\n"
                    source += (
                        f"### Human:\n{data['input']}\n### Assistant:\n"
                    )
                    self.sources.append(source)

                    target = f"{data['output']}{tokenizer.eos_token}"

                    self.targets.append(target)
                # else:
                # dialog mode

        del data_list
        gc.collect()

        logging.warning("there are {} data in dataset".format(len(self.sources)))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(42)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        )
    
    # lora config
    lora_modules = [
        "q_proj",
        # "k_proj",
        "v_proj",
        "o_proj",
        # "up_proj",
        # "gate_proj",
        # "down_proj",
    ]
    # lora_modules = "all-linear"
    
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()