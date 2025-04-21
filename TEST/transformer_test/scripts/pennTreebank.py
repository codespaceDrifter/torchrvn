from datasets import load_dataset
import os

# Get current file's directory and construct the target path
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(os.path.dirname(current_dir), "datasets")
print(f"Dataset directory: {dataset_dir}")

os.makedirs(dataset_dir, exist_ok=True)

ptb_dataset = load_dataset("ptb_text_only", trust_remote_code=True)
ptb_dataset.save_to_disk(os.path.join(dataset_dir, "ptb"))