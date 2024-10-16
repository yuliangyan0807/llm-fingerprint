from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import os
import torch

# model_name_or_path = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_deepseek_8_test"
# model_name_or_path = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_deepseek_8"
model_name_or_path = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_llama3_8_test"
# model_name_or_path = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models"


# model_name_or_path = "/mnt/data/yuliangyan/deepseek-ai/deepseek-llm-7b-base/"

# def save_hf_format(model, tokenizer, args, sub_folder=""):
#     # used to save huggingface format, so we can use it for hf.from_pretrained
#     model_to_save = model.module if hasattr(model, 'module') else model
#     CONFIG_NAME = "config.json"
#     WEIGHTS_NAME = "pytorch_model.bin"
#     # output_dir = os.path.join(args.output_dir, sub_folder)
#     output_dir = "/home/yuliangyan/Code/llm-fingerprinting/stanford_alpaca/saved_models_deepseek_8_test"
#     os.makedirs(output_dir, exist_ok=True)
#     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
#     output_config_file = os.path.join(output_dir, CONFIG_NAME)
#     save_dict = model_to_save.state_dict()
#     for key in list(save_dict.keys()):
#         if "lora" in key:
#             del save_dict[key]
#     torch.save(save_dict, output_model_file)
#     model_to_save.config.to_json_file(output_config_file)
#     # tokenizer.save_vocabulary(output_dir)
#     tokenizer.save_pretrained(output_config_file)


config = PeftConfig.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                            return_dict=True, 
                                            device_map="auto",
                                            output_hidden_states=True
                                            )
peft_model = PeftModel.from_pretrained(model, model_name_or_path, 
                                            return_dict=True, 
                                            device_map="auto",
                                            output_hidden_states=True
                                            )
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
print(peft_model)

# peft_config = LoraConfig(
#     task_type="CAUSAL_LM", 
#     inference_mode=False, 
#     r=8, 
#     lora_alpha=32, 
#     lora_dropout=0.1
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# save_hf_format(model, tokenizer, "debug",)