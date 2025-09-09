from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import torch

print("Starting tokenize_datasets.py")
# load tokenizer
bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")

# Set special tokens
bpe_tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>"
})

print("Loaded tokenizer with vocab size:", len(bpe_tokenizer))
# read 6 datasets from ../datasets - en and de of train test valdiation
def read_lines(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    if len(src_lines) != len(tgt_lines):
        print(f"Warning: {src_path} has {len(src_lines)} lines, {tgt_path} has {len(tgt_lines)} lines")

    filtered_src, filtered_tgt = [], []
    for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        src = src.rstrip("\n")
        tgt = tgt.rstrip("\n")
        if src != "" and tgt != "":
            filtered_src.append(src)
            filtered_tgt.append(tgt)
        else:
            # optional: print first few mismatches
            if len(filtered_src) < 5:
                print(f"Skipping empty/mismatched line {idx}: EN='{src}' DE='{tgt}'")

    print(f"Read {len(filtered_src)} aligned non-empty lines from {src_path} and {tgt_path}")
    return filtered_src, filtered_tgt

train_en, train_de = read_lines("datasets/train.en", "datasets/train.de")
val_en, val_de = read_lines("datasets/validation.en", "datasets/validation.de")
test_en, test_de = read_lines("datasets/test.en", "datasets/test.de")


print("Line counts after reading:")
print("  Train EN:", len(train_en), " Train DE:", len(train_de))
print("  Val   EN:", len(val_en),  " Val   DE:", len(val_de))
print("  Test  EN:", len(test_en),  " Test  DE:", len(test_de))

# Debug: check for empty lines
empty_train_en = sum(1 for l in train_en if l.strip() == "")
empty_train_de = sum(1 for l in train_de if l.strip() == "")
print("Empty lines -> Train EN:", empty_train_en, " Train DE:", empty_train_de)

# Show example mismatched lines if counts differ
if len(train_en) != len(train_de):
    print("!!! Train EN/DE mismatch detected !!!")
    for i, (en, de) in enumerate(zip(train_en, train_de)):
        if en.strip() == "" or de.strip() == "":
            print(f"  Line {i}: EN=<{en}> DE=<{de}>")
            break  # only show first mismatch

# use bpe_tokenizer.json tokenizer to tokenise, adding sepcial tokens pad, eos, sos, returning pt
def tokenize_sequences(sequences, tokenizer, max_len):
    return tokenizer(sequences, padding = "max_length", max_length = max_len, return_tensors = "pt", add_special_tokens=True, truncation = True)

# vocab size : 37000

# return src  and tgt seq length
def calculate_seq_len(lines,tokenizer):
    return max(len(tokenizer(line).input_ids) for line in lines)
"""
src_seq_len = calculate_seq_len(train_en + val_en + test_en, bpe_tokenizer)  # English
tgt_seq_len = calculate_seq_len(train_de + val_de + test_de, bpe_tokenizer)  # German

# print(src_seq_len) # 14517
# print(tgt_seq_len) # 9691
"""

src_seq_len = 350
tgt_seq_len = 350