import sys
import os
import torch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from utils.formatting.txt2bin import tokenize_to_dataset
from utils.model.tokenizer import Tokenizer

tokenizer = Tokenizer.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/tokenizer.json")))

tokenize_to_dataset(
    tokenizer=tokenizer,
    input_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/openwebtext.txt")),
    train_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/openwebtext_train.bin")),
    test_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/openwebtext_test.bin")),
    valid_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/openwebtext_valid.bin")),
    train_ratio=0.8,
    test_ratio=0.1,
    valid_ratio=0.1,
    start_line=0,
    dtype=torch.int32
)

