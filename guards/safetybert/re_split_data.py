import json
import random
import os

input_file = "/data/liyc/SafetyBERT/data/converted_data.jsonl"
train_file = "/data/liyc/SafetyBERT/data/sample_train.jsonl"
valid_file = "/data/liyc/SafetyBERT/data/sample_valid.jsonl"
test_file = "/data/liyc/SafetyBERT/data/sample_test.jsonl"

def re_split():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)
    
    train_lines = lines[:train_end]
    valid_lines = lines[train_end:valid_end]
    test_lines = lines[valid_end:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line)
            
    with open(valid_file, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line)

    with open(test_file, 'w', encoding='utf-8') as f:
        for line in test_lines:
            f.write(line)
            
    print(f"Total: {total}")
    print(f"Saved {len(train_lines)} to {train_file} (80%)")
    print(f"Saved {len(valid_lines)} to {valid_file} (10%)")
    print(f"Saved {len(test_lines)} to {test_file} (10%)")

if __name__ == "__main__":
    re_split()
