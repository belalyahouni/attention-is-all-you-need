"""
Data Preprocessing Module
========================

This module handles all data loading and preprocessing tasks for the Transformer model.
It provides functions to:
1. Load parallel text datasets (English-German)
2. Filter and clean the data
3. Prepare data for tokenization
4. Load the trained BPE tokenizer

This module is essential for preparing the training data in the correct format
for the Transformer model.

Author: Implementation for Transformer training pipeline
"""

from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import torch

print("Starting data_preprocessing.py")

# ============================================================================
# TOKENIZER LOADING
# ============================================================================

# Load the pre-trained BPE tokenizer
bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")

# Add special tokens required for the model
bpe_tokenizer.add_special_tokens({
    "pad_token": "<pad>",    # Padding token for batch processing
    "bos_token": "<bos>",    # Beginning of sequence token
    "eos_token": "<eos>"    # End of sequence token
})

print("Loaded tokenizer with vocab size:", len(bpe_tokenizer))

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def read_lines(src_path, tgt_path):
    """
    Read parallel text files and return aligned sentence pairs.
    
    This function reads source and target language files line by line,
    ensuring that each line corresponds to a translation pair.
    It also filters out empty lines and mismatched pairs.
    
    Args:
        src_path (str): Path to source language file
        tgt_path (str): Path to target language file
        
    Returns:
        tuple: (filtered_source_lines, filtered_target_lines)
        
    Note:
        The function ensures both files have the same number of lines
        and filters out any empty or mismatched pairs.
    """
    # Read both files simultaneously
    with open(src_path, "r", encoding="utf-8") as f_src, \
         open(tgt_path, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    # Check for length mismatch
    if len(src_lines) != len(tgt_lines):
        print(f"Warning: {src_path} has {len(src_lines)} lines, {tgt_path} has {len(tgt_lines)} lines")

    # Filter out empty lines and mismatched pairs
    filtered_src, filtered_tgt = [], []
    for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        # Remove newline characters and check for empty content
        src = src.rstrip("\n")
        tgt = tgt.rstrip("\n")
        
        # Only keep non-empty pairs
        if src != "" and tgt != "":
            filtered_src.append(src)
            filtered_tgt.append(tgt)
        else:
            # Optional: print first few mismatches for debugging
            if len(filtered_src) < 5:
                print(f"Skipping empty/mismatched line {idx}: EN='{src}' DE='{tgt}'")

    print(f"Read {len(filtered_src)} aligned non-empty lines from {src_path} and {tgt_path}")
    return filtered_src, filtered_tgt

# ============================================================================
# DATASET LOADING
# ============================================================================

# Load all dataset splits: training, validation, and test
print("Loading dataset files...")
train_en, train_de = read_lines("datasets/train.en", "datasets/train.de")
val_en, val_de = read_lines("datasets/validation.en", "datasets/validation.de")
test_en, test_de = read_lines("datasets/test.en", "datasets/test.de")

# Print dataset statistics
print("\nDataset Statistics:")
print("  Training EN:", len(train_en), " Training DE:", len(train_de))
print("  Validation EN:", len(val_en), " Validation DE:", len(val_de))
print("  Test EN:", len(test_en), " Test DE:", len(test_de))

# Check for empty lines in training data\empty_train_en = sum(1 for l in train_en if l.strip() == "")
empty_train_de = sum(1 for l in train_de if l.strip() == "")
print("\nEmpty lines check:")
print("  Training EN:", empty_train_en, " Training DE:", empty_train_de)

# Debug: Show mismatched lines if counts differ
if len(train_en) != len(train_de):
    print("\n!!! Training EN/DE mismatch detected !!!")
    for i, (en, de) in enumerate(zip(train_en, train_de)):
        if en.strip() == "" or de.strip() == "":
            print(f"  Line {i}: EN=<{en}> DE=<{de}>")
            break  # Only show first mismatch

# ============================================================================
# TOKENIZATION UTILITIES
# ============================================================================

def tokenize_sequences(sequences, tokenizer, max_len):
    """
    Tokenize a list of text sequences using the BPE tokenizer.
    
    This function converts raw text into token IDs that the model can process.
    It handles padding and truncation to ensure all sequences have the same length.
    
    Args:
        sequences: List of text strings to tokenize
        tokenizer: PreTrainedTokenizerFast instance
        max_len: Maximum sequence length (sequences will be padded/truncated to this)
        
    Returns:
        Dictionary containing tokenized sequences with input_ids
        
    Note:
        Special tokens (<bos>, <eos>, <pad>) are automatically added
    """
    return tokenizer(
        sequences, 
        padding="max_length", 
        max_length=max_len, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True
    )

# ============================================================================
# SEQUENCE LENGTH CONFIGURATION
# ============================================================================

# Note: Originally calculated dynamically, but set to fixed values for consistency
# This ensures all sequences are padded/truncated to the same length for batch processing

# Fixed sequence lengths (used throughout training)
src_seq_len = 350  # Maximum source sequence length
tgt_seq_len = 350  # Maximum target sequence length

print(f"\nSequence length configuration:")
print(f"  Source max length: {src_seq_len}")
print(f"  Target max length: {tgt_seq_len}")
print(f"  Vocabulary size: {len(bpe_tokenizer)}")

# ============================================================================
# DATA LOADING FUNCTION FOR TRAINING
# ============================================================================

def load_and_prepare_data():
    """
    Load and prepare datasets for training.
    
    This function loads the training and validation datasets, ensuring they are
    properly aligned and filtered. It's designed to be called once during
    training initialization to avoid reloading data multiple times.
    
    Returns:
        tuple: (train_en, train_de, val_en, val_de) - Lists of aligned sentences
        
    Note:
        This function should only be called once per training session
        to avoid unnecessary file I/O operations.
    """
    print("--- Loading and preparing datasets (this should only run once) ---")
    
    # Load training and validation data
    train_en, train_de = read_lines("datasets/train.en", "datasets/train.de")
    val_en, val_de = read_lines("datasets/validation.en", "datasets/validation.de")
    
    print("\nDataset loading complete:")
    print("  Training EN:", len(train_en), " Training DE:", len(train_de))
    print("  Validation EN:", len(val_en), " Validation DE:", len(val_de))

    return train_en, train_de, val_en, val_de