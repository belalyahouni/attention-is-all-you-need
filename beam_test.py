import torch
import torch.nn.functional as F
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
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    })
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Model hyperparameters (must match training)
vocab_size = bpe_tokenizer.vocab_size
src_seq_len = 350
tgt_seq_len = 350

# Rebuild model from a single checkpoint
transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)
checkpoint_path = "checkpoints/step_100000.pt" # Update with your checkpoint
print(f"Loading model from {checkpoint_path}...")

checkpoint = torch.load(checkpoint_path, map_location=device)
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.to(device)
transformer.eval()

# Token IDs
bos_token_id = bpe_tokenizer.bos_token_id
eos_token_id = bpe_tokenizer.eos_token_id
pad_token_id = bpe_tokenizer.pad_token_id

# ----------------------------
# 2. Translate function with Beam Search
# ----------------------------
def translate_beam_search(
    src_sentence: str,
    beam_size: int = 4,
    length_penalty_alpha: float = 0.6,
    max_output_len: int = 150 # A sensible default max length
):
    """
    Translates a source sentence using beam search with a length penalty.
    """
    with torch.no_grad():
        # 1. Encode the source sentence
        encoder_input = bpe_tokenizer(
            src_sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=src_seq_len,
            truncation=True
        )["input_ids"].to(device)

        enc_mask = create_encoder_mask(encoder_input, pad_token_id)
        enc_out = transformer.encode(encoder_input, enc_mask)

        # 2. Initialize beams
        # A beam consists of (sequence, score). Start with the BOS token.
        initial_beam = (torch.tensor([[bos_token_id]], dtype=torch.long, device=device), 0.0)
        beams = [initial_beam]
        completed_beams = []

        # 3. Decoding loop
        for _ in range(max_output_len):
            new_beams = []
            for seq, score in beams:
                # If a beam ended with EOS, it's complete. Don't expand it.
                if seq[0, -1].item() == eos_token_id:
                    completed_beams.append((seq, score))
                    continue

                # Get model predictions for the next token
                dec_mask = create_decoder_mask(seq, pad_token_id)
                dec_out = transformer.decode(enc_out, enc_mask, seq, dec_mask)
                logits = transformer.project(dec_out)
                
                # Get log probabilities for the last token in the sequence
                next_token_log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get the top `beam_size` next tokens and their log probabilities
                top_k_log_probs, top_k_tokens = torch.topk(next_token_log_probs, beam_size, dim=-1)

                # Create new candidate beams by extending the current beam
                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_k_tokens[:, i].unsqueeze(0)], dim=1)
                    new_score = score + top_k_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))

            # 4. Prune beams
            # Sort all candidate beams by score and keep only the top `beam_size`
            if not new_beams:
                break # All beams might have completed
                
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # If all top beams are completed, we can stop early
            if all(b[0][0, -1].item() == eos_token_id for b in beams):
                completed_beams.extend(beams)
                break
        
        # 5. Select the best completed beam
        # Add any remaining active beams to the completed list
        completed_beams.extend(beams)
        
        # Apply length penalty to find the best translation
        # Score formula: score / (sequence_length ** alpha)
        if not completed_beams: # Handle case with no completed beams
            return ""

        best_beam = max(
            completed_beams,
            key=lambda x: x[1] / (x[0].size(1) ** length_penalty_alpha) if x[0].size(1) > 0 else -1e9
        )
        
        # Decode the final sequence
        output_ids = best_beam[0].squeeze(0)[1:] # Exclude BOS token
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
    # Call the new beam search function. The parameters match the paper's description.
    translated_sentence = translate_beam_search(
        src_sentence,
        beam_size=4,
        length_penalty_alpha=0.6
    )
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
