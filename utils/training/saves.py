import os
import re
import torch

def cleanup_checkpoints(folder, keep_last_n=5):
    def extract_epoch_batch(filename):
        match = re.search(r"epoch_(\d+)_batch_(\d+)", filename)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2))
            return (epoch, batch)
        return (-1, -1)

    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    files_sorted = sorted(files, key=extract_epoch_batch, reverse=True)  # newest first

    for f in files_sorted[keep_last_n:]:
        os.remove(os.path.join(folder, f))

def load_latest_checkpoint(folder, model):
    def extract_epoch_batch(filename):
        match = re.search(r"epoch_(\d+)_batch_(\d+)", filename)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2))
            return (epoch, batch)
        return 0, 0

    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if not files:
        return 0, 0  # Start from beginning if no checkpoints

    files_sorted = sorted(files, key=extract_epoch_batch, reverse=True)  # newest first
    
    # Try loading latest checkpoint first
    try:
        latest_file = files_sorted[0]
        print(f"Loading latest checkpoint: {latest_file}")
        model.load_state_dict(torch.load(os.path.join(folder, latest_file)))
        return extract_epoch_batch(latest_file)
    except Exception as e:
        print(f"Error loading latest checkpoint: {e}")
        if len(files_sorted) < 2:
            return 0, 0  # Not enough checkpoints to try second latest
        second_latest_file = files_sorted[1]
        print(f"Loading second latest checkpoint: {second_latest_file}")
        model.load_state_dict(torch.load(os.path.join(folder, second_latest_file)))
        return extract_epoch_batch(second_latest_file)


