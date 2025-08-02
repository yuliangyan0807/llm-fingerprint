import os
from openai import OpenAI
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import time
import json

# Set your OpenAI API key
client = OpenAI(
    base_url="your_base_url",
    api_key="your_api_key"
)

def load_trajectory_dataset():
    """Load the trajectory dataset from disk"""
    print("Loading trajectory dataset...")
    dataset = load_from_disk('./data/trajectory_set')
    print(f"Loaded {len(dataset)} samples")
    return dataset

def paraphrase_with_gpt(text, template="paraphrase"):
    """
    Use GPT to paraphrase the given text using different templates
    
    Args:
        text (str): The text to paraphrase
        template (str): The paraphrase template to use
    
    Returns:
        str: The paraphrased text
    """
    templates = {
        "paraphrase": "Please paraphrase the following text while maintaining the same meaning and tone:",
        # "formal": "Please rewrite the following text in a more formal academic style while preserving the core meaning:",
        # "casual": "Please rewrite the following text in a more casual, conversational style while keeping the same meaning:",
        # "technical": "Please rewrite the following text using more technical language while maintaining the same meaning:",
        # "simple": "Please rewrite the following text using simpler language while preserving the meaning:"
    }
    
    prompt = f"{templates.get(template, templates['paraphrase'])}\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases text while maintaining the original meaning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=64,
            temperature=1.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error paraphrasing text: {e}")
        return text  # Return original text if paraphrasing fails

def process_dataset(dataset, template="paraphrase", save_interval=1000):
    """
    Process the dataset by paraphrasing the decode_output field
    
    Args:
        dataset: The input dataset
        template (str): The paraphrase template to use
        save_interval (int): Save progress every N samples
    
    Returns:
        Dataset: The processed dataset with paraphrased decode_output
    """
    print(f"Processing dataset with template: {template}")
    print(f"Saving progress every {save_interval} samples")
    
    output_path = f"./data/trajectory_set_{template}"
    temp_path = f"{output_path}_temp"
    
    # Check if there's a temporary save file to resume from
    start_index = 0
    processed_data = []
    
    if os.path.exists(temp_path):
        print(f"Found temporary save file at {temp_path}")
        response = input("Do you want to resume from the temporary save? (y/n): ")
        if response.lower() == 'y':
            temp_dataset = load_from_disk(temp_path)
            processed_data = list(temp_dataset)
            start_index = len(processed_data)
            print(f"Resuming from sample {start_index}")
        else:
            print("Starting fresh...")
    
    for i in tqdm(range(start_index, len(dataset))):
        # Get the full sample as a dictionary
        sample = dataset[i]
        
        # Create a copy of the sample
        new_sample = sample.copy()
        
        # Paraphrase the decode_output field
        original_output = sample['decode_output']
        paraphrased_output = paraphrase_with_gpt(original_output, template)
        
        # Update the decode_output field
        new_sample['decode_output'] = paraphrased_output
        
        processed_data.append(new_sample)
        
        # Save progress periodically
        if (i + 1) % save_interval == 0:
            print(f"\nSaving progress at sample {i + 1}...")
            temp_dataset = Dataset.from_list(processed_data)
            temp_dataset.save_to_disk(temp_path)
            print(f"Progress saved to {temp_path}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
    
    return Dataset.from_list(processed_data)

def save_dataset(dataset, output_path):
    """Save the processed dataset to disk"""
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Dataset saved successfully!")

def main():
    """Main function to run the paraphrase attack"""
    
    # Check if client is set
    if not client:
        print("Please set your OpenAI API key by uncommenting and setting the api_key in the script")
        return
    
    # Load the original dataset
    dataset = load_trajectory_dataset()
    
    # Process with paraphrase template
    template = "paraphrase"
    save_interval = 1000  # Save every 1000 samples
    
    print(f"\n{'='*50}")
    print(f"Processing with template: {template}")
    print(f"Save interval: {save_interval} samples")
    print(f"{'='*50}")
    
    # Process the dataset
    processed_dataset = process_dataset(dataset, template=template, save_interval=save_interval)
    
    # Save the processed dataset
    output_path = f"./data/trajectory_set_{template}"
    save_dataset(processed_dataset, output_path)
    
    # Show some examples
    print(f"\nExamples of paraphrased outputs ({template}):")
    for i in range(min(3, len(processed_dataset))):
        print(f"Original: {dataset[i]['decode_output'][:100]}...")
        print(f"Paraphrased: {processed_dataset[i]['decode_output'][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()