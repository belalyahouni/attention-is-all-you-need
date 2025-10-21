"""
Transformer Model Implementation
===============================

This module contains the complete implementation of the Transformer architecture
from the "Attention Is All You Need" paper. It includes all core components:
- Layer normalization
- Feed-forward networks
- Input embeddings
- Positional encoding
- Multi-head attention
- Encoder and decoder blocks
- Complete transformer model

Author: Implementation based on Vaswani et al. (2017)
"""

import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    """
    Layer Normalization Component
    
    This implements layer normalization as described in the Transformer paper.
    It normalizes the input across the feature dimension to help with training
    stability and convergence.
    
    Args:
        features (int): The number of features (embedding dimension)
        eps (float): Small value to prevent division by zero (default: 1e-6)
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma in the paper)
        self.alpha = nn.Parameter(torch.ones(features))
        # Learnable shifting parameter (beta in the paper)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Calculate mean and standard deviation across the feature dimension
        # keepdim=True preserves dimensions for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # Shape: (batch, seq_len, 1)
        
        # Apply normalization formula: (x - mean) / (std + eps)
        # Add small epsilon to prevent division by zero
        normalized = (x - mean) / (std + self.eps)
        
        # Apply learnable scaling and shifting parameters
        return self.alpha * normalized + self.bias

class FeedForwardBlock(nn.Module):
    """
    Feed-Forward Network Block
    
    This implements the position-wise feed-forward network from the Transformer paper.
    It consists of two linear transformations with a ReLU activation in between.
    The first layer expands the dimension, the second contracts it back.
    
    Args:
        d_model (int): Model dimension (embedding size)
        d_ff (int): Feed-forward dimension (usually 4 * d_model)
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # First linear transformation: expand dimension
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # Second linear transformation: contract back to original dimension
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply first linear layer, ReLU activation, dropout, then second linear layer
        # Shape transformation: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    """
    Input Embedding Layer
    
    Converts token indices to dense vector representations (embeddings).
    The embeddings are scaled by sqrt(d_model) as specified in the Transformer paper.
    
    Args:
        d_model (int): Embedding dimension
        vocab_size (int): Size of the vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Learnable embedding matrix: each token gets a d_model-dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Convert token indices to embeddings.
        
        Args:
            x: Token indices tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedding tensor of shape (batch_size, seq_len, d_model)
        """
        # Convert token indices to embeddings and scale by sqrt(d_model)
        # This scaling helps with training stability
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding Layer
    
    Adds positional information to input embeddings using sinusoidal functions.
    This allows the model to understand the order of tokens in the sequence,
    since the Transformer has no inherent notion of position.
    
    Args:
        d_model (int): Model dimension (must match embedding dimension)
        seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the divisor for the sinusoidal functions
        # This creates different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding added
        """
        # Add positional encoding to input embeddings
        # Only use the first x.shape[1] positions (in case sequence is shorter than max)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization
    
    Implements the "Add & Norm" component from the Transformer paper.
    This applies layer normalization to the input, passes it through a sublayer,
    adds the result to the original input (residual connection), and applies dropout.
    
    Args:
        features (int): Number of features (embedding dimension)
        dropout (float): Dropout probability
    """
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection with layer normalization.
        
        Args:
            x: Input tensor
            sublayer: Function to apply (e.g., attention or feed-forward)
            
        Returns:
            Output after residual connection and normalization
        """
        # Apply layer norm, then sublayer, then add residual connection
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block
    
    Implements the scaled dot-product attention mechanism with multiple attention heads.
    This allows the model to attend to different types of relationships simultaneously.
    
    Args:
        d_model (int): Model dimension (embedding size)
        h (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h              # Number of attention heads
        
        # Ensure d_model is divisible by number of heads
        assert d_model % h == 0, "d_model must be divisible by h"

        # Dimension of vector seen by each head
        self.d_k = d_model // h
        
        # Linear transformations for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query transformation
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key transformation
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value transformation
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output transformation
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch, heads, seq_len, d_k)
            key: Key tensor of shape (batch, heads, seq_len, d_k)
            value: Value tensor of shape (batch, heads, seq_len, d_k)
            mask: Attention mask (optional)
            dropout: Dropout layer
            
        Returns:
            Tuple of (attention_output, attention_scores)
        """
        d_k = query.shape[-1]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to very negative value)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        
        # Apply dropout to attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Compute weighted sum of values
        attention_output = attention_scores @ value
        
        return attention_output, attention_scores

    def forward(self, q, k, v, mask):
        """
        Apply multi-head attention.
        
        Args:
            q: Query tensor of shape (batch, seq_len, d_model)
            k: Key tensor of shape (batch, seq_len, d_model)
            v: Value tensor of shape (batch, seq_len, d_model)
            mask: Attention mask (optional)
            
        Returns:
            Attention output of shape (batch, seq_len, d_model)
        """
        # Apply linear transformations to get Q, K, V
        query = self.w_q(q)  # Shape: (batch, seq_len, d_model)
        key = self.w_k(k)    # Shape: (batch, seq_len, d_model)
        value = self.w_v(v)  # Shape: (batch, seq_len, d_model)

        # Reshape and transpose for multi-head attention
        # Split d_model into h heads, each with d_k dimensions
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Now shape: (batch, h, seq_len, d_k)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Combine all heads together
        # Transpose back and reshape to original dimensions
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Shape: (batch, seq_len, d_model)

        # Apply final linear transformation
        return self.w_o(x)

class EncoderBlock(nn.Module):
    """
    Single Encoder Block
    
    A complete encoder block containing:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    
    Args:
        features (int): Number of features (embedding dimension)
        self_attention_block: Multi-head attention block
        feed_forward_block: Feed-forward network block
        dropout (float): Dropout probability
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections: one for attention, one for feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Source mask to ignore padding tokens
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        x = self.residual_connections[0](x, 
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Complete Encoder
    
    Stacks multiple encoder blocks and applies final layer normalization.
    The encoder processes the input sequence and creates rich representations
    that capture relationships between all input tokens.
    
    Args:
        features (int): Number of features (embedding dimension)
        layers: List of encoder blocks
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Forward pass through all encoder blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Source mask to ignore padding tokens
            
        Returns:
            Encoded representations of shape (batch, seq_len, d_model)
        """
        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final layer normalization
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Single Decoder Block
    
    A complete decoder block containing:
    1. Masked multi-head self-attention with residual connection
    2. Multi-head cross-attention with residual connection
    3. Feed-forward network with residual connection
    
    Args:
        features (int): Number of features (embedding dimension)
        self_attention_block: Masked self-attention block
        cross_attention_block: Cross-attention block
        feed_forward_block: Feed-forward network block
        dropout (float): Dropout probability
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Three residual connections: self-attention, cross-attention, feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            encoder_output: Output from encoder
            src_mask: Source mask to ignore padding tokens
            tgt_mask: Target mask for causal attention
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Masked self-attention with residual connection
        x = self.residual_connections[0](x, 
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # Cross-attention with residual connection
        x = self.residual_connections[1](x, 
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # Feed-forward with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    """
    Complete Decoder
    
    Stacks multiple decoder blocks and applies final layer normalization.
    The decoder generates output tokens one by one, using both:
    - Self-attention on previously generated tokens (with causal masking)
    - Cross-attention to the encoder output
    
    Args:
        features (int): Number of features (embedding dimension)
        layers: List of decoder blocks
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through all decoder blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            encoder_output: Output from encoder
            src_mask: Source mask to ignore padding tokens
            tgt_mask: Target mask for causal attention
            
        Returns:
            Decoded representations of shape (batch, seq_len, d_model)
        """
        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply final layer normalization
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Output Projection Layer
    
    Final linear layer that projects decoder output to vocabulary size.
    This converts the model's internal representations to logits over
    the vocabulary, which can then be used to predict the next token.
    
    Args:
        d_model (int): Model dimension (embedding size)
        vocab_size (int): Size of the vocabulary
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        """
        Project decoder output to vocabulary logits.
        
        Args:
            x: Decoder output of shape (batch, seq_len, d_model)
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.proj(x)
    
class Transformer(nn.Module):
    """
    Complete Transformer Model
    
    The full Transformer architecture combining encoder and decoder.
    This is the main model class that orchestrates the entire
    encoding and decoding process.
    
    Args:
        encoder: Encoder component
        decoder: Decoder component
        src_embed: Source language embeddings
        tgt_embed: Target language embeddings
        src_pos: Source positional encoding
        tgt_pos: Target positional encoding
        projection_layer: Output projection layer
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encode source sequence.
        
        Args:
            src: Source token indices of shape (batch, seq_len)
            src_mask: Source mask to ignore padding tokens
            
        Returns:
            Encoded representations of shape (batch, seq_len, d_model)
        """
        # Convert tokens to embeddings
        src = self.src_embed(src)
        # Add positional encoding
        src = self.src_pos(src)
        # Pass through encoder
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decode target sequence.
        
        Args:
            encoder_output: Output from encoder
            src_mask: Source mask to ignore padding tokens
            tgt: Target token indices of shape (batch, seq_len)
            tgt_mask: Target mask for causal attention
            
        Returns:
            Decoded representations of shape (batch, seq_len, d_model)
        """
        # Convert tokens to embeddings
        tgt = self.tgt_embed(tgt)
        # Add positional encoding
        tgt = self.tgt_pos(tgt)
        # Pass through decoder
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Project decoder output to vocabulary logits.
        
        Args:
            x: Decoder output of shape (batch, seq_len, d_model)
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, 
                       tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, 
                       dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Build a complete Transformer model.
    
    This function creates all the components of the Transformer architecture
    and assembles them into a complete model. It follows the original paper's
    specifications for the base model.
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        src_seq_len (int): Maximum source sequence length
        tgt_seq_len (int): Maximum target sequence length
        d_model (int): Model dimension (default: 512)
        N (int): Number of encoder/decoder layers (default: 6)
        h (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
        d_ff (int): Feed-forward dimension (default: 2048)
        
    Returns:
        Complete Transformer model
    """
    # Create embedding layers for source and target
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, 
                                   feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                   decoder_cross_attention_block, 
                                   feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the complete transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, 
                            src_pos, tgt_pos, projection_layer)
    
    # Initialize parameters using Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer