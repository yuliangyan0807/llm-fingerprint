import os
import json
import random
subject_list = ["biology"]

def get_res_list(res):
    """Extracts predictions for correctly answered questions."""
    rlt = []
    # random.seed(12345)  # Ensure reproducibility
    
    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                rlt.append(each["question_id"])
        elif each["pred"] == each["answer"]:
            rlt.append(each["question_id"])
    return rlt

def read_answers(model_path):
    """Reads model-generated answers from JSON files."""
    categories = subject_list  # Ensure subject_list is defined elsewhere
    data = {}
    
    for category in categories:
        output_path = os.path.join(model_path, f"{category}.json")
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                    data[category] = [e for e in entries if 'question_id' in e]
            except json.JSONDecodeError as e:
                print(f"Error in {output_path}: {e}")
    return data

def extract_answers(output_path, num=20):
    """Extracts answers from multiple models."""
    rlt = {}
    model_name = "Meta-Llama-3.1-8B-Instruct"
    base_path = os.path.join(output_path, model_name)
    sampled_indices = sample_questions(base_path, num)
    
    if os.path.exists(base_path):
        data = read_answers(base_path)
        rlt[model_name] = {}
        
        for category in subject_list:
            try:
                extract = [e for e in data.get(category, []) if e['question_id'] in sampled_indices.get(category, [])]
                rlt[model_name][category] = [e['pred'] for e in extract]
            except KeyError:
                print(f"Error processing category {category} in {model_name}")
    return rlt

def sample_questions(default_model_file, sample_size):
    """Samples questions randomly while avoiding masked IDs."""
    seed = random.randint(0, 2**32 - 1)  # Randomized seed
    data = read_answers(default_model_file)
    random.seed(seed)
    sampled_indices = {}
    
    for category in subject_list:
        num_questions = len(data.get(category, []))
        remaining_indices = [i for i in range(num_questions) if data[category][i]['question_id'] not in MaskIds]
        actual_sample_size = min(sample_size, len(remaining_indices))
        sampled_question_ids = [data[category][idx]['question_id'] for idx in random.sample(remaining_indices, actual_sample_size)]
        if actual_sample_size > 0:
            sampled_indices[category] = sampled_question_ids
    return sampled_indices

def save_res(res, output_path):
    """Saves results to a JSON file and computes accuracy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fo:
        json.dump(res, fo)
    
    correct, incorrect = 0, 0
    for each in res:
        if not each["pred"]:
            if random.randint(0, len(each["options"]) - 1) == each["answer_index"]:
                correct += 1
            else:
                incorrect += 1
        elif each["pred"] == each["answer"]:
            correct += 1
        else:
            incorrect += 1
    
    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    return accuracy, correct, incorrect
def select_by_category(df, subject):
    """Filters dataset by category."""
    return [each for each in df if each["category"] == subject]

def select_by_indicies(df, sampled_indices):
    """Selects entries by given indices."""
    return [each for each in df if each["question_id"] in sampled_indices]
def load_similarity_data():
    """Loads similarity data from JSON files."""
    try:
        with open('./Knowledge_model_similarities.json', 'r', encoding='utf-8') as f:
            knowledge_similarity = json.load(f)
        with open('./base_model_similarities.json', 'r', encoding='utf-8') as f:
            base_similarity = json.load(f)
        return knowledge_similarity, base_similarity
    except FileNotFoundError:
        print("Error: Similarity data files not found.")
        return {}, {}
def load_final_mask(filepath="final_mask.json"):
    """
    Read final_mask, return set type
    """
    with open(filepath, "r", encoding="utf-8") as f:
        mask_list = json.load(f)
        return set(mask_list)