"""
Model Evaluation Script
=====================

This script evaluates a trained Transformer model on the test dataset
and calculates BLEU scores for machine translation quality assessment.

Key features:
- Load trained model from checkpoint
- Greedy decoding for translation
- BLEU score calculation using SacreBLEU
- Command-line interface for checkpoint selection
- Progress tracking during evaluation

Usage:
    python model_evaluation.py <checkpoint_filename>
    
Example:
    python model_evaluation.py step_300000.pt

Requirements:
    - Pre-trained tokenizer (bpe_tokenizer.json)
    - Test dataset files in datasets/ directory
    - Model checkpoint in checkpoints/ directory

Author: Implementation for Transformer evaluation pipeline
"""

import torch
import sys
import os
from transformers import PreTrainedTokenizerFast
import sacrebleu
from transformer_model import build_transformer
from attention_masks import create_encoder_mask, create_decoder_mask

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BPE tokenizer
try:
    bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")
    bpe_tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<bos>",  # Beginning of sequence
        "eos_token": "<eos>"  # End of sequence
    })
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Model hyperparameters (must match training configuration)
vocab_size = bpe_tokenizer.vocab_size
src_seq_len = 350  # Maximum source sequence length
tgt_seq_len = 350  # Maximum target sequence length

print(f"Model configuration:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Source max length: {src_seq_len}")
print(f"  Target max length: {tgt_seq_len}")

# ============================================================================
# MODEL LOADING
# ============================================================================

# Rebuild the Transformer model
print("Building Transformer model...")
transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

# Check command-line arguments
if len(sys.argv) < 2:
    print("Error: No checkpoint file specified.")
    print("Usage: python3 model_evaluation.py <checkpoint_filename>")
    print("Example: python3 model_evaluation.py step_300000.pt")
    exit()

# Get checkpoint filename from command line
checkpoint_filename = sys.argv[1]
checkpoint_path = os.path.join("checkpoints", checkpoint_filename)

# Verify checkpoint file exists
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
    exit()

print(f"Loading model from {checkpoint_path}...")

# Load checkpoint with error handling
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Extract model state dict (new checkpoint format)
    transformer.load_state_dict(checkpoint['model_state_dict'])
except KeyError:
    print(f"Error: 'model_state_dict' key not found in checkpoint file.")
    print("Please ensure the checkpoint was saved correctly.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# Move model to device and set to evaluation mode
transformer.to(device)
transformer.eval()
print("Model loaded successfully!")

# Cache special token IDs for efficiency
bos_token_id = bpe_tokenizer.bos_token_id
eos_token_id = bpe_tokenizer.eos_token_id
pad_token_id = bpe_tokenizer.pad_token_id

# ============================================================================
# TRANSLATION FUNCTION
# ============================================================================

def translate(src_sentence, max_output_len=100):
    """
    Translate a source sentence using greedy decoding.
    
    This function performs greedy decoding, which means it always selects
    the token with the highest probability at each step. While not optimal,
    it's fast and often produces reasonable translations.
    
    Args:
        src_sentence (str): Source sentence to translate
        max_output_len (int): Maximum length of output sequence
        
    Returns:
        str: Translated sentence
    """
    with torch.no_grad():  # Disable gradient computation for inference
        # Tokenize source sentence
        encoder_input = bpe_tokenizer(
            src_sentence, 
            return_tensors="pt",
            padding="max_length", 
            max_length=src_seq_len,
            truncation=True
        )["input_ids"].to(device)

        # Initialize decoder with beginning-of-sequence token
        decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

        # Generate tokens one by one
        for _ in range(max_output_len):
            # Create attention masks
            enc_mask = create_encoder_mask(encoder_input, pad_token_id)
            dec_mask = create_decoder_mask(decoder_input, pad_token_id)
            
            # Forward pass through the model
            enc_out = transformer.encode(encoder_input, enc_mask)
            dec_out = transformer.decode(enc_out, enc_mask, decoder_input, dec_mask)
            logits = transformer.project(dec_out)

            # Greedy decoding: select token with highest probability
            next_token = logits[:, -1, :].argmax(-1).unsqueeze(0)
            
            # Append predicted token to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Stop if end-of-sequence token is generated
            if next_token.item() == eos_token_id:
                break
        
        # Convert token IDs back to text (exclude the initial <bos> token)
        output_ids = decoder_input.squeeze(0)[1:]
        output_text = bpe_tokenizer.decode(output_ids.cpu().tolist(), skip_special_tokens=True)
        return output_text

# ============================================================================
# TEST DATA LOADING
# ============================================================================

def load_data(file_path):
    """
    Load text data from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of non-empty lines from the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Load test datasets
print("Loading test datasets...")
test_source_sentences = load_data('datasets/test.en')
test_target_sentences = load_data('datasets/test.de')

print(f"Loaded {len(test_source_sentences)} test sentences.")
print(f"Sample source: {test_source_sentences[0] if test_source_sentences else 'N/A'}")
print(f"Sample target: {test_target_sentences[0] if test_target_sentences else 'N/A'}")

# ============================================================================
# TRANSLATION AND EVALUATION
# ============================================================================

print("\nStarting translation and evaluation...")
print("=" * 50)

# Lists to store generated translations and references
generated_translations = []
# SacreBLEU expects references as list of lists
references = [[ref] for ref in test_target_sentences]

# Translate each test sentence
for i, src_sentence in enumerate(test_source_sentences):
    # Translate the source sentence
    translated_sentence = translate(src_sentence)
    generated_translations.append(translated_sentence)
    
    # Print progress and sample translations
    if i % 100 == 0:
        print(f"\nProgress: {i+1}/{len(test_source_sentences)} sentences translated")
        if i < len(test_target_sentences):
            print(f"  Source: '{src_sentence}'")
            print(f"  Target: '{test_target_sentences[i]}'")
            print(f"  Model:  '{translated_sentence}'")
            print("-" * 50)

# ============================================================================
# BLEU SCORE CALCULATION
# ============================================================================

print("\nCalculating BLEU score...")
print("=" * 50)

# Calculate BLEU score on the entire test corpus
bleu_score = sacrebleu.corpus_bleu(generated_translations, references)

# Print final results
print(f"\nEvaluation Results:")
print(f"  Test sentences: {len(test_source_sentences)}")
print(f"  Final Corpus BLEU score: {bleu_score.score:.2f}")
print("=" * 50)