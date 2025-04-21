from datasets import load_dataset
import os

# Get current file's directory and construct the target path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(os.path.dirname(current_dir), "datasets")
print(f"Dataset directory: {dataset_dir}")

os.makedirs(dataset_dir, exist_ok=True)

wiki_dataset = load_dataset("wikitext", "wikitext-2-v1")
wiki_dataset.save_to_disk(os.path.join(dataset_dir, "wiki"))