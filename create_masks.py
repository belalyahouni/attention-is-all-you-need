import torch

def create_encoder_mask(input_ids, pad_token_id):
    """
    Creates a mask for the encoder to ignore padding tokens.
    Input is a 1D tensor of token IDs for a single sequence.
    Output shape: (1, 1, seq_len)
    """
    return (input_ids != pad_token_id).unsqueeze(0).unsqueeze(0)

def create_decoder_mask(seq, pad_token_id):
    """
    Creates a combined mask for the decoder.
    It masks both padding tokens and subsequent tokens (for causality).
    Input is a 1D tensor of token IDs for a single sequence.
    Output shape: (1, seq_len, seq_len)
    """
    # Get the sequence length from the 1D tensor
    seq_len = seq.size(0)

    # 1. Create a padding mask to ignore pad tokens.
    # Shape: (1, 1, seq_len)
    padding_mask = (seq != pad_token_id).unsqueeze(0).unsqueeze(0)

    # 2. Create a subsequent mask to prevent looking ahead.
    # This creates a lower-triangular matrix.
    # Shape: (seq_len, seq_len)
    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()

    # 3. Combine them. PyTorch's broadcasting will make the shapes compatible.
    # (1, 1, seq_len) & (seq_len, seq_len) -> (1, seq_len, seq_len)
    final_mask = padding_mask & subsequent_mask
    return final_mask
