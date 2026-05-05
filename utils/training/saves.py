import os
import re
import json
import torch


def extract_batch(filename):
    match = re.search(r"batch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def cleanup_checkpoints(folder, keep_last_n=5):
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    files_sorted = sorted(files, key=extract_batch, reverse=True)  # newest first

    for f in files_sorted[keep_last_n:]:
        os.remove(os.path.join(folder, f))


def save_checkpoint(path, model, optimizer, metadata, scheduler=None):
    # atomic save - writes .tmp then renames
    temp_path = path + ".tmp"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)

    # metadata dict goes to json (caller decides what to put in it)
    folder = os.path.dirname(path)
    meta_dir = os.path.join(folder, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    json_path = os.path.join(meta_dir, "training_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)


def load_latest_checkpoint(folder, model, optimizer=None, scheduler=None):
    # returns (metadata_dict,) or None if no checkpoint found
    files = [f for f in os.listdir(folder) if f.endswith(".pt") and not f.endswith(".tmp")]
    if not files:
        return None

    files_sorted = sorted(files, key=extract_batch, reverse=True)

    # load metadata if exists
    json_path = os.path.join(folder, "metadata", "training_metadata.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)
    else:
        # no metadata json - just extract batch from filename
        metadata = {'batch': extract_batch(files_sorted[0])}

    # load weights (try newest, fallback to second newest)
    for checkpoint_file in files_sorted[:2]:
        try:
            path = os.path.join(folder, checkpoint_file)
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(path)

            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            return metadata

        except Exception as e:
            print(f"Error loading {checkpoint_file}: {e}")
            continue

    return None
