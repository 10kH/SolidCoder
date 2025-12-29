import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def prepare_code_contest():
    output_path = "data/CodeContest/Test.jsonl"
    if os.path.exists(output_path):
        print(f"CodeContest data already exists at {output_path}")
        return

    print("Downloading CodeContest dataset (test split)...")
    try:
        # Load from HuggingFace
        dataset = load_dataset("deepmind/code_contests", split="test")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    print("Converting to CodeGenerator format...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    formatted_data = []
    
    for item in tqdm(dataset):
        # Helper to format test cases
        def format_io(inputs, outputs):
            return [{"input": inp, "output": [out]} for inp, out in zip(inputs, outputs)]

        public_tests = format_io(item['public_tests']['input'], item['public_tests']['output'])
        private_tests = format_io(item['private_tests']['input'], item['private_tests']['output'])
        generated_tests = format_io(item['generated_tests']['input'], item['generated_tests']['output'])
        
        # Combine private and generated tests for evaluation
        all_tests = private_tests + generated_tests
        
        formatted_item = {
            "name": item['name'],
            "description": item['description'],
            "tags": item['cf_tags'],
            "difficulty": item['difficulty'],
            "id": item['cf_contest_id'],
            "source": item['source'],
            "sample_io": public_tests,
            "test_list": all_tests
        }
        formatted_data.append(formatted_item)

    print(f"Saving {len(formatted_data)} items to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
    print("Done!")

if __name__ == "__main__":
    prepare_code_contest()
