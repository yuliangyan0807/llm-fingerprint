import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

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
                                                        # device_map="auto",
                                                        output_hidden_states=True,
                                                        # torch_dtype=torch.bfloat16,
                                                        # quantization_config=bnb_config
                                                        )
            model = PeftModel.from_pretrained(model, model_name_or_path, 
                                                        return_dict=True, 
                                                        # device_map="auto",
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