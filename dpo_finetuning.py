import os
import torch
import transformers
from peft import PeftModel, PeftConfig

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from utils import *

def return_prompt_and_responses(samples):
    return {
        "prompt": [
            f"An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```{input}```\n ### Output: "
            for input in samples["question"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
# class TrainingArguments(transformers.TrainingArguments):
class TrainingArguments(DPOConfig):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="paged_adamw_32bit"
        # default='adamw_torch'
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    max_length: int = field(
        default=2048,
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

if __name__ == '__main__':
    # output_dir="./dpo_results"
    # model_name = "merged_peft/final_merged_checkpoint"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    # model.config.use_cache = False
    
    # config = PeftConfig.from_pretrained(model_args.model_name_or_path, 
    #                                     cache_dir=training_args.cache_dir,
    #                                     )
    # base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
    #                                     cache_dir=training_args.cache_dir,
    #                                             )
    # model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path, 
    #                                     cache_dir=training_args.cache_dir,
    #                                             )
    # tokenizer = AutoTokenizer.from_pretrained(
    #                                         model_args.model_name_or_path,
    #                                         cache_dir=training_args.cache_dir,
    #                                         model_max_length=3096,
    #                                         padding_side="right",
    #                                         use_fast=False,
    #                                         trust_remote_code=True,
    #                                         )
    model, tokenizer = load_hf_model(model_name_or_path=model_args.model_name_or_path,
                                     bnb_config=bnb_config)
    
    model = prepare_model_for_kbit_training(model)

    # model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

    dataset = load_dataset(data_args.data_path, split="train")
    dataset = dataset.select(range(100))
    original_columns = dataset.column_names

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )

    # training_args = TrainingArguments(
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     gradient_checkpointing =True,
    #     max_grad_norm= 0.3,
    #     num_train_epochs=15, 
    #     save_steps= 100,
    #     learning_rate=2e-4,
    #     bf16=True,
    #     save_total_limit=3,
    #     logging_steps=10,
    #     output_dir=output_dir,
    #     optim="paged_adamw_32bit",
    #     lr_scheduler_type="cosine",
    #     warmup_ratio=0.05,
    #     remove_unused_columns=False
    # )

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        # target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dpo_trainer = DPOTrainer(
        model,
        # model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )

    dpo_trainer.train()
    # dpo_trainer.save_model(output_dir)

    # output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)