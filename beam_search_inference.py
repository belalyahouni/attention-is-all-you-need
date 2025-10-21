"""
Beam Search Inference Script
===========================

This script implements beam search decoding for the Transformer model,
providing higher quality translations compared to greedy decoding.

Key features:
- Beam search with configurable beam size
- Length penalty for better translation quality
- Complete test set evaluation
- BLEU score calculation
- Progress tracking and sample outputs

Beam search explores multiple translation paths simultaneously,
keeping track of the most promising candidates. This often results
in better translations than greedy decoding.

Usage:
    python beam_search_inference.py

Requirements:
    - Pre-trained tokenizer (bpe_tokenizer.json)
    - Test dataset files in datasets/ directory
    - Model checkpoint in checkpoints/ directory

Author: Implementation for Transformer evaluation pipeline
"""

import torch
import torch.nn.functional as F
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
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    })
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Model hyperparameters (must match training configuration)
vocab_size = bpe_tokenizer.vocab_size
src_seq_len = 350
tgt_seq_len = 350

print(f"Model configuration:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Source max length: {src_seq_len}")
print(f"  Target max length: {tgt_seq_len}")

# ============================================================================
# MODEL LOADING
# ============================================================================

# Rebuild the Transformer model
transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)

# Load model checkpoint
checkpoint_path = "checkpoints/step_100000.pt"  # Update with your checkpoint
print(f"Loading model from {checkpoint_path}...")

checkpoint = torch.load(checkpoint_path, map_location=device)
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.to(device)
transformer.eval()
print("Model loaded successfully!")

# Cache special token IDs for efficiency
bos_token_id = bpe_tokenizer.bos_token_id
eos_token_id = bpe_tokenizer.eos_token_id
pad_token_id = bpe_tokenizer.pad_token_id

# ============================================================================
# BEAM SEARCH TRANSLATION FUNCTION
# ============================================================================

def translate_beam_search(
    src_sentence: str,
    beam_size: int = 4,
    length_penalty_alpha: float = 0.6,
    max_output_len: int = 150
):
    """
    Translate a source sentence using beam search with length penalty.
    
    Beam search maintains multiple translation candidates simultaneously,
    exploring different paths through the vocabulary. At each step, it
    keeps the most promising candidates based on their cumulative scores.
    
    Args:
        src_sentence (str): Source sentence to translate
        beam_size (int): Number of beams to maintain (default: 4)
        length_penalty_alpha (float): Length penalty factor (default: 0.6)
        max_output_len (int): Maximum output sequence length (default: 150)
        
    Returns:
        str: Best translation found by beam search
        
    Note:
        Length penalty helps prevent the model from generating
        overly short or long translations.
    """
    with torch.no_grad():
        # ========================================================================
        # ENCODE SOURCE SENTENCE
        # ========================================================================
        
        encoder_input = bpe_tokenizer(
            src_sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=src_seq_len,
            truncation=True
        )["input_ids"].to(device)

        # Create encoder mask and encode
        enc_mask = create_encoder_mask(encoder_input, pad_token_id)
        enc_out = transformer.encode(encoder_input, enc_mask)

        # ========================================================================
        # INITIALIZE BEAM SEARCH
        # ========================================================================
        
        # Each beam is a tuple: (sequence_tensor, cumulative_score)
        # Start with the beginning-of-sequence token
        initial_beam = (torch.tensor([[bos_token_id]], dtype=torch.long, device=device), 0.0)
        beams = [initial_beam]
        completed_beams = []  # Beams that have generated <eos> token

        # ========================================================================
        # BEAM SEARCH DECODING LOOP
        # ========================================================================
        
        for _ in range(max_output_len):
            new_beams = []
            
            # Process each active beam
            for seq, score in beams:
                # Skip beams that have already completed (ended with <eos>)
                if seq[0, -1].item() == eos_token_id:
                    completed_beams.append((seq, score))
                    continue

                # ================================================================
                # GET MODEL PREDICTIONS
                # ================================================================
                
                # Create decoder mask and get model predictions
                dec_mask = create_decoder_mask(seq, pad_token_id)
                dec_out = transformer.decode(enc_out, enc_mask, seq, dec_mask)
                logits = transformer.project(dec_out)
                
                # Get log probabilities for the last token in the sequence
                next_token_log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get top-k most likely next tokens
                top_k_log_probs, top_k_tokens = torch.topk(
                    next_token_log_probs, beam_size, dim=-1
                )

                # ================================================================
                # CREATE NEW CANDIDATE BEAMS
                # ================================================================
                
                # Extend current beam with each top-k token
                for i in range(beam_size):
                    # Create new sequence by appending token
                    new_seq = torch.cat([seq, top_k_tokens[:, i].unsqueeze(0)], dim=1)
                    # Update cumulative score
                    new_score = score + top_k_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))

            # ====================================================================
            # PRUNE BEAMS
            # ====================================================================
            
            # If no new beams were created, break
            if not new_beams:
                break
                
            # Sort all candidate beams by score and keep only the top beam_size
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # Early stopping: if all top beams are completed, we can stop
            if all(b[0][0, -1].item() == eos_token_id for b in beams):
                completed_beams.extend(beams)
                break
        
        # ========================================================================
        # SELECT BEST TRANSLATION
        # ========================================================================
        
        # Add any remaining active beams to completed beams
        completed_beams.extend(beams)
        
        # Handle case with no completed beams
        if not completed_beams:
            return ""

        # Apply length penalty to find the best translation
        # Formula: score / (sequence_length ** alpha)
        # This penalizes very short or very long sequences
        best_beam = max(
            completed_beams,
            key=lambda x: x[1] / (x[0].size(1) ** length_penalty_alpha) 
                         if x[0].size(1) > 0 else -1e9
        )
        
        # Convert token IDs back to text (exclude the initial <bos> token)
        output_ids = best_beam[0].squeeze(0)[1:]
        output_text = bpe_tokenizer.decode(
            output_ids.cpu().tolist(), 
            skip_special_tokens=True
        )
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

# ============================================================================
# BEAM SEARCH TRANSLATION AND EVALUATION
# ============================================================================

print("\nStarting beam search translation and evaluation...")
print("=" * 60)

# Lists to store generated translations and references
generated_translations = []
# SacreBLEU expects references as list of lists
references = [[ref] for ref in test_target_sentences]

# Translate each test sentence using beam search
for i, src_sentence in enumerate(test_source_sentences):
    # Use beam search with parameters from the original paper
    translated_sentence = translate_beam_search(
        src_sentence,
        beam_size=4,              # Number of beams
        length_penalty_alpha=0.6  # Length penalty factor
    )
    generated_translations.append(translated_sentence)
    
    # Print progress and sample translations
    if i % 100 == 0:
        print(f"\nProgress: {i+1}/{len(test_source_sentences)} sentences translated")
        if i < len(test_target_sentences):
            print(f"  Source: '{src_sentence}'")
            print(f"  Target: '{test_target_sentences[i]}'")
            print(f"  Model:  '{translated_sentence}'")
            print("-" * 60)

# ============================================================================
# BLEU SCORE CALCULATION
# ============================================================================

print("\nCalculating BLEU score...")
print("=" * 60)

# Calculate BLEU score on the entire test corpus
bleu_score = sacrebleu.corpus_bleu(generated_translations, references)

# Print final results
print(f"\nBeam Search Evaluation Results:")
print(f"  Test sentences: {len(test_source_sentences)}")
print(f"  Beam size: 4")
print(f"  Length penalty: 0.6")
print(f"  Final Corpus BLEU score: {bleu_score.score:.2f}")
print("=" * 60)
