import torch
from ..model.tokenizer import Tokenizer
import os

def tokenize_to_dataset(tokenizer: Tokenizer,
                        input_path: str,
                        train_path: str,
                        test_path: str,
                        valid_path: str,
                        train_ratio: float = 0.8,
                        test_ratio: float = 0.1,
                        valid_ratio: float = 0.1,
                        start_line: int = 0,
                        dtype: torch.dtype = torch.int32):
    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First pass: just count lines
    print("Counting lines...")
    total_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                total_lines += 1
    
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)
    
    # Second pass: process one line at a time
    processed_lines = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as in_file, \
         open(train_path, "ab") as train_file, \
         open(test_path, "ab") as test_file, \
         open(valid_path, "ab") as valid_file:
        
        for line in in_file:
            line = line.strip()

            if not line:
                continue

            if processed_lines < start_line:
                processed_lines += 1
                continue
                
            # Process single line
            ids = tokenizer.encode(line, add_SOS=False, add_EOS=False).tolist()
            arr = torch.tensor(ids, dtype=dtype).numpy().tobytes()
            
            # Write to appropriate split
            if processed_lines < train_end:
                train_file.write(arr)
            elif processed_lines < test_end:
                test_file.write(arr)
            else:
                valid_file.write(arr)
                
            processed_lines += 1
            if processed_lines % 1000 == 0:
                print(f"Processed {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")
                print(f"Line: {line}")
                print(f"IDs: {ids}")
        
        print(f"✅ Done. Split into {train_end} train, {test_end-train_end} test, {total_lines-test_end} valid examples")