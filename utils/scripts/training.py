import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.training.train import train_model
from utils.model.tokenizer import Tokenizer
from algorithms.transformer.classic_transformer import Transformer
from utils.dataset.unstrucuted_text_dataset import TransformerDataset
from torch import nn
import torch

train_dataset = TransformerDataset(
    path=os.path.join(project_root, "TEST/datasets/openwebtext_train.bin"),
    input_len=64,
    output_len=32,
    stride=96,
    dtype=torch.int32
)

test_dataset = TransformerDataset(
    path=os.path.join(project_root, "TEST/datasets/openwebtext_test.bin"),
    input_len=64,
    output_len=32,
    stride=96,
    dtype=torch.int32)

valid_dataset = TransformerDataset(
    path=os.path.join(project_root, "TEST/datasets/openwebtext_valid.bin"),
    input_len=64,
    output_len=32,
    stride=96,
    dtype=torch.int32)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
print(f"Valid dataset length: {len(valid_dataset)}")
print(f"Train dataset sample: {train_dataset[0]}")

tokenizer = Tokenizer.load(os.path.join(project_root, "TEST/tokenizer/web_text_tokenizer.json"))

first_20 = {k: tokenizer.id2token[k] for k in list(tokenizer.id2token.keys())[:20]}
print(first_20)

transformer = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_encoders=4,
    num_decoders=4,
    d_ff=2048)

train_model(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=transformer,
    optimizer=torch.optim.Adam(transformer.parameters(), lr=0.0001),
    batch_size=16,
    num_epochs=10,
    save_folder_path=os.path.join(project_root, "TEST/checkpoints"),
    batch_per_save=100
)

# command to monitor gpu cuda core and vram usage
# nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv --loop=1


