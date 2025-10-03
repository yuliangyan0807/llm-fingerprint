import time
import logging
import json
import torch
from tqdm import tqdm
from vllm.lora.request import LoRARequest
from datasets import load_dataset  # Load dataset module
from Utils import  get_res_list


def preprocess(test_df):
    """Preprocess dataset by filtering out invalid options."""
    res_df = []
    for each in test_df:
        options = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = options
        res_df.append(each)
    return res_df

def generate_cot_prompt(val_df, curr, k):
    """Generates a chain-of-thought prompt from validation examples."""
    prompt = ""
    with open(f"./initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]

    val_df = val_df[: k]  # Select top-k examples
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt

def format_cot_example(example, including_answer=True):
    """Formats a single example for CoT prompting."""
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def select_by_indicies(df, sampled_indices):
    """Selects rows based on provided indices."""
    res = []
    for each in df:
        if each["question_id"] in sampled_indices:
            res.append(each)
    return res

def select_by_category(df, subject):
    """Selects rows based on category."""
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res
