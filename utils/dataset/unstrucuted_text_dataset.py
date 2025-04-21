import torch
import torch.utils.data as data
import numpy as np
import os
from typing import List, Tuple, Dict, Optional

class TransformerDataset(data.Dataset):
    def __init__(self, path: str,
                 input_len: int,
                 output_len: int,
                 stride: int = None,
                 SOS_token: int = 0,
                 EOS_token: int = 1,
                 dtype: torch.dtype = torch.int32):
        self.path = path
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride if stride is not None else input_len
        self.dtype = dtype 
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

        self.item_size = (input_len + output_len) * torch.tensor([], dtype=dtype).element_size()
        self.total_tokens = os.path.getsize(path) // torch.tensor([], dtype=dtype).element_size()
        self.length = (self.total_tokens - (input_len + output_len)) // stride

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        byte_offset = idx * self.stride * torch.tensor([], dtype=self.dtype).element_size()
        with open(self.path, "rb") as f:
            f.seek(byte_offset)
            buf = f.read(self.item_size)
            tokens = torch.frombuffer(buf, dtype=self.dtype)

        # Input sequence: take full input_len and add SOS
        x = torch.cat([
            tokens[:self.input_len],
            torch.tensor([self.EOS_token], dtype=self.dtype)
        ])

        # Target sequence: take full output_len and add SOS + EOS
        y = torch.cat([
            torch.tensor([self.SOS_token], dtype=self.dtype),
            tokens[self.input_len:self.input_len + self.output_len],
            torch.tensor([self.EOS_token], dtype=self.dtype)
        ])

        return x, y