import torch
from transformers import PreTrainedTokenizerFast
from transformer_model import build_transformer
from attention_masks import create_encoder_mask, create_decoder_mask
import os

# ----------------------------
# 1. Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")
# Set special tokens
bpe_tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>"
})
# Model hyperparameters (must match training)
vocab_size = bpe_tokenizer.vocab_size
src_seq_len = 350   # source max length used in training
tgt_seq_len = 350    # target max length used in training

# Rebuild model
transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)

# Load latest checkpoint
checkpoint_path = "checkpoints/step_90000.pt"  # change if needed
transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    # Tokenize source
    encoder_input = bpe_tokenizer(src_sentence, return_tensors="pt",
                                  padding="max_length", max_length=src_seq_len,
                                  truncation=True)["input_ids"].to(device)

    # Initialize decoder input with <bos>
    decoder_input = torch.tensor([[bos_token_id]], device=device)

    for _ in range(max_output_len):
        # Create masks
        enc_mask = create_encoder_mask(encoder_input, pad_token_id)
        dec_mask = create_decoder_mask(decoder_input, pad_token_id)

        # Forward pass
        enc_out = transformer.encode(encoder_input, enc_mask)
        dec_out = transformer.decode(enc_out, enc_mask, decoder_input, dec_mask)
        logits = transformer.projection(dec_out)

        # Greedy decoding: pick token with highest probability
        next_token = logits[:, -1, :].argmax(-1).unsqueeze(0)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        # Stop if <eos> token generated
        if next_token.item() == eos_token_id:
            break

    # Decode output tokens to text
    output_ids = decoder_input[0, 1:]  # remove <bos>
    output_text = bpe_tokenizer.decode(output_ids.tolist())
    return output_text

# ----------------------------
# 3. Run translation
# ----------------------------
input_sentence = "Resumption of the session"
output_sentence = translate(input_sentence)
print("Input:", input_sentence)
print("Output:", output_sentence)
