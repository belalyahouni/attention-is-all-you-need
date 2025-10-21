import torch
import sys
import os
from transformers import PreTrainedTokenizerFast
import sacrebleu
from model import build_transformer
from create_masks import create_encoder_mask, create_decoder_mask

# ----------------------------
# 1. Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
try:
    bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")
    bpe_tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<bos>", # Using <bos> for consistency as it's often used as SOS
        "eos_token": "<eos>"
    })
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Model hyperparameters (must match training)
vocab_size = bpe_tokenizer.vocab_size
src_seq_len = 350
tgt_seq_len = 350

# Rebuild model
transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)

# --- KEY CHANGE: LOAD CHECKPOINT FROM NEW DICTIONARY STRUCTURE ---
# checkpoint_path = "checkpoints/step_360000.pt" # Update with your final checkpoint

# --- MODIFIED SECTION: LOAD CHECKPOINT FROM COMMAND-LINE ARGUMENT ---

# 1. Check if a command-line argument was provided
if len(sys.argv) < 2:
    print("Error: No checkpoint file specified.")
    print("Usage: python3 new_test.py <checkpoint_filename>")
    print("Example: python3 new_test.py step_300000.pt")
    exit()

# 2. Get the filename from the command line and construct the full path
checkpoint_filename = sys.argv[1]
checkpoint_path = os.path.join("checkpoints", checkpoint_filename)

# 3. Check if the file exists before attempting to load
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
    exit()

print(f"Loading model from {checkpoint_path}...")

# Load the entire checkpoint dictionary
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Extract the model's state_dict using the 'model_state_dict' key
    transformer.load_state_dict(checkpoint['model_state_dict'])
except KeyError:
    print(f"Error: 'model_state_dict' key not found in the checkpoint file. Please ensure it was saved correctly.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()
# -----------------------------------------------------------------

transformer.to(device)
transformer.eval()

# Token IDs
bos_token_id = bpe_tokenizer.bos_token_id
eos_token_id = bpe_tokenizer.eos_token_id
pad_token_id = bpe_tokenizer.pad_token_id

# ----------------------------
# 2. Translate function
# ----------------------------
def translate(src_sentence, max_output_len=100):
    with torch.no_grad(): # Inference should be done without gradient calculation
        encoder_input = bpe_tokenizer(src_sentence, return_tensors="pt",
                                      padding="max_length", max_length=src_seq_len,
                                      truncation=True)["input_ids"].to(device)

        # Start decoding with the BOS token
        decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_output_len):
            enc_mask = create_encoder_mask(encoder_input, pad_token_id)
            dec_mask = create_decoder_mask(decoder_input, pad_token_id)
            
            enc_out = transformer.encode(encoder_input, enc_mask)
            dec_out = transformer.decode(enc_out, enc_mask, decoder_input, dec_mask)
            logits = transformer.project(dec_out)

            # Greedy decoding: select the token with the highest probability
            next_token = logits[:, -1, :].argmax(-1).unsqueeze(0)
            
            # Append the predicted token to the decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Stop if the EOS token is generated
            if next_token.item() == eos_token_id:
                break
        
        # Exclude the initial BOS token from the final output
        output_ids = decoder_input.squeeze(0)[1:]
        output_text = bpe_tokenizer.decode(output_ids.cpu().tolist(), skip_special_tokens=True)
        return output_text

# ----------------------------
# 3. Load Test Data
# ----------------------------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

test_source_sentences = load_data('datasets/test.en')
test_target_sentences = load_data('datasets/test.de')

print(f"Loaded {len(test_source_sentences)} test sentences.")

# ----------------------------
# 4. Generate and Evaluate Translations
# ----------------------------
generated_translations = []
# Create a list of lists for references, as SacreBLEU expects this format
references = [[ref] for ref in test_target_sentences]

for i, src_sentence in enumerate(test_source_sentences):
    translated_sentence = translate(src_sentence)
    generated_translations.append(translated_sentence)
    
    # Optional: Print progress and a sample translation
    if i % 100 == 0:
        print(f"Translating sentence {i+1}/{len(test_source_sentences)}...")
        if i < len(test_target_sentences):
          print(f"  Source: '{src_sentence}'")
          print(f"  Target: '{test_target_sentences[i]}'")
          print(f"  Model:  '{translated_sentence}'")
          print("-" * 20)

# Calculate BLEU score on the entire corpus
print("\nCalculating BLEU score...")
bleu_score = sacrebleu.corpus_bleu(generated_translations, references)

# Print final result
print(f"Final Corpus BLEU score: {bleu_score.score:.2f}")