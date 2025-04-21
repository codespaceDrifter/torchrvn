import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.training.train import train_model
from utils.model.tokenizer import Tokenizer
from algorithms.transformer.classic_transformer import Transformer
from utils.dataset.unstrucuted_text_dataset import TransformerDataset
from utils.training.saves import load_latest_checkpoint
from utils.inference.inference import inference
from torch import nn
import torch

tokenizer = Tokenizer.load(os.path.join(project_root, "TEST/tokenizer/web_text_tokenizer.json"))

first_20 = {k: tokenizer.id2token[k] for k in list(tokenizer.id2token.keys())[:20]}

transformer = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_encoders=4,
    num_decoders=4,
    d_ff=2048)

print(transformer)

load_latest_checkpoint(os.path.join(project_root, "TEST/checkpoints"), transformer)

inference(transformer, tokenizer)


