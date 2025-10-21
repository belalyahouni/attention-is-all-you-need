"""
Attention Masks Module
=====================

This module provides functions to create attention masks for the Transformer model.
Attention masks are crucial for:
1. Ignoring padding tokens in the input sequences
2. Preventing the decoder from looking at future tokens (causal masking)

The masks ensure that the model only attends to relevant positions during
both self-attention and cross-attention operations.

Author: Implementation based on Vaswani et al. (2017)
"""

import torch


def create_encoder_mask(input_ids, pad_token_id):
    """
    Create attention mask for encoder to ignore padding tokens.
    
    The encoder mask prevents the model from attending to padding tokens,
    which are added to make all sequences in a batch the same length.
    Without this mask, the model would learn to attend to meaningless
    padding tokens, degrading performance.
    
    Args:
        input_ids: 1D tensor of token IDs for a single sequence
        pad_token_id: ID of the padding token
        
    Returns:
        Mask tensor of shape (1, 1, seq_len) where True means "attend" and False means "ignore"
        
    Example:
        >>> input_ids = torch.tensor([1, 2, 3, 0, 0])  # 0 is padding
        >>> mask = create_encoder_mask(input_ids, pad_token_id=0)
        >>> print(mask)
        tensor([[[True, True, True, False, False]]])
    """
    # Create boolean mask: True for real tokens, False for padding tokens
    # unsqueeze operations add dimensions for broadcasting with attention scores
    return (input_ids != pad_token_id).unsqueeze(0).unsqueeze(0)

def create_decoder_mask(seq, pad_token_id):
    """
    Create combined attention mask for decoder.
    
    The decoder mask combines two types of masking:
    1. Padding mask: Ignores padding tokens (same as encoder)
    2. Causal mask: Prevents looking at future tokens (lower triangular matrix)
    
    This ensures that during training, the decoder can only attend to:
    - Previous tokens in the sequence (causal constraint)
    - Non-padding tokens (padding constraint)
    
    Args:
        seq: 1D tensor of token IDs for a single sequence
        pad_token_id: ID of the padding token
        
    Returns:
        Combined mask tensor of shape (1, seq_len, seq_len)
        where True means "attend" and False means "ignore"
        
    Example:
        >>> seq = torch.tensor([1, 2, 3, 0, 0])  # 0 is padding
        >>> mask = create_decoder_mask(seq, pad_token_id=0)
        >>> print(mask.shape)
        torch.Size([1, 5, 5])
        # The mask will be lower triangular AND ignore padding positions
    """
    # Get sequence length from input tensor
    seq_len = seq.size(0)

    # 1. Create padding mask: True for real tokens, False for padding
    # Shape: (1, 1, seq_len) - will broadcast to (1, seq_len, seq_len)
    padding_mask = (seq != pad_token_id).unsqueeze(0).unsqueeze(0)

    # 2. Create causal mask: Lower triangular matrix
    # This prevents attending to future tokens
    # Shape: (seq_len, seq_len)
    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()

    # 3. Combine both masks using logical AND
    # Broadcasting: (1, 1, seq_len) & (seq_len, seq_len) -> (1, seq_len, seq_len)
    # Result: True only where both padding_mask AND subsequent_mask are True
    final_mask = padding_mask & subsequent_mask
    
    return final_mask
