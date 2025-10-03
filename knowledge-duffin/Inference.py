import time
import logging
import json
import torch
from tqdm import tqdm
from vllm.lora.request import LoRARequest
from datasets import load_dataset  # Load dataset module
from Utils import extract_answer, get_res_list


def batch_inference(llm, sampling_params, inference_batch):
    """Performs batch inference using VLM."""
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(f"Batch size {len(inference_batch)} processed in {time.time() - start} sec")
    
    response_batch, pred_batch = [], []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append("MASKED")  # Masked for open-source compliance
        pred_batch.append("MASKED")
    return pred_batch, response_batch

def batch_inference_adapter(llm, sampling_params, inference_batch, lora_path):
    """Performs batch inference with LoRA adapter."""
    start = time.time()
    outputs = llm.generate(
        inference_batch, sampling_params,
        lora_request=LoRARequest("lora_adapter", 1, lora_path)
    )
    logging.info(f"Batch size {len(inference_batch)} processed in {time.time() - start} sec")
    
    response_batch, pred_batch = [], []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append("MASKED")
        pred_batch.append("MASKED")
    return pred_batch, response_batch

def batch_generation(model, tokenizer, prompt, max_new_tokens=1024):
    """Generates text using the model in batch mode."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=True).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
    
    model.eval()
    generation_input = {
        "input_ids": input_ids,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    
    with torch.no_grad():
        output = model.generate(**generation_input)
    
    gen_sequences = output.sequences[:, input_ids.shape[-1]:]
    try:
        decoded_output = [tokenizer.decode(ids) for ids in gen_sequences]
    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        decoded_output = ["" for _ in range(len(gen_sequences))]
    
    response_batch, pred_batch = ["MASKED"] * len(decoded_output), ["MASKED"] * len(decoded_output)
    return pred_batch, response_batch

def eval_cot_sample(subject, model, tokenizer, val_df, test_df, output_path, is_save, is_adapter=False, adapter_path=None):
    """Evaluates model performance using Chain-of-Thought (CoT) prompting."""
    llm, sampling_params = model
    logging.info(f"Evaluating {subject}")
    inference_batches = []
    
    for i in tqdm(range(len(test_df))):
        k, prompt_length_ok = 5, False
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, test_df[i], k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            if len(inputs["input_ids"][0]) < 4096 - 1024:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)
    
    pred_batch, response_batch = (batch_inference_adapter(llm, sampling_params, inference_batches, adapter_path)
                                  if is_adapter else batch_inference(llm, sampling_params, inference_batches))
    
    res = [{"pred": "MASKED", "model_outputs": "MASKED"} for _ in test_df]
    
    if is_save:
        with open(output_path, "w") as fo:
            json.dump(res, fo)
    
    return get_res_list(res)
