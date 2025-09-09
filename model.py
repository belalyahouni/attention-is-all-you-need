import torch
import torch.nn as nn
import math

"""
Class for input embeddings:
takes input tokens and uses embedding layer to convert to vectors, then scale with root of d_model
"""

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)
        return out

"""
Class for positional encoding:
calculate positional encoding, assign each input vector (token) a positional encoding using sin/cos
add to the encoding
apply dropout

creates vectors of d_model size for every inoput vector
all inpout vectors will be padded to maximum sequence length, so use seq_len
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create empty tensor to use for base of positional encoding
        pe = torch.zeros(seq_len, d_model)


        # create a 1d tensor from 0 to max sequence length, then add a dimension
        # stores the position of the input embedding, top of positoinal encoding formula
        # shape: seq_len, 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # formula for bottom of positonal encoding formula
        # 10000^2i/d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sin formula to even and cos formula to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add dimension for batch of sequences
        # shape: 1 (batch), seq_len, d_model
        pe = pe.unsqueeze(0)

        # store positional encoding weights
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add x to positiobnal encoding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # apply dropout
        out = self.dropout(x)
        return out


""" 
class LayerNorm(nn.Module):
    # calc mean and variance of each item in btach independtenlty, calc new values for each using own mean + variance.
    # gamma and beta (mulitcative and additive) allow module posibility to amplify )
    # aim: normlaise input so each feature has zero mean and unit variance per training example
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        # trainable parameter
        self.alpha = nn.Parameter(torch.ones(1)) # multipies
        self.bias = nn.Parameter(torch.zeros(1)) # addition

    def forward(self, x):
        # take x and normalise it using an normalisation equation
        # calc mean and std
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim= True)
        
        out = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return out
"""

# feed forward block aplpying linear, relu, and linear then a dropout
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout= nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        assert (self.d_k * heads == d_model), "d_model must be divisible by heads" 

        # 3 big lienar transformations, later split into 8 heads
        self.wq = nn.Linear(d_model, d_model, bias= False)
        self.wk = nn.Linear(d_model, d_model, bias= False)
        self.wv = nn.Linear(d_model, d_model, bias= False)

        self.wo = nn.Linear(d_model, d_model, bias= False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):

        # calculate vectors of 512 for query key and value (all heads combined)
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        # split into heads, reshaping
        # before: batch (shape0), seq len (shape1), d_model
        # split d_model dimension into d_model/head=d_k,head
        # after: batch, seq len, heads, dk
        # after transpose: batch, heads, seq len, dk

        # transpose because mmul expects heads to follow batch
        query = query.reshape(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2)
        key = key.reshape(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2)
        value = value.reshape(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2)

        # now we have split into heads, ready for formula

        # q x k.transpose and divide by root d_k
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(self.d_k)

        # mask before softmax, so pr0obabilty isnt based onc ertain embeddings
        # if mask, set mask (minus inf)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e20)

        # calculate probalityies with softmax
        attention_scores = attention_scores.softmax(dim = -1)

        
        attention_scores = self.dropout(attention_scores)

        x = attention_scores @ value

        # combine all ehads together

        # reverse the transpose (seq len in 2nd position), concatinating all heads, making it one things again
        # combines the two dimensiosn that were split earlier
        x = x.transpose(1,2).reshape(x.shape[0], -1, self.heads * self.d_k)

        # final linear transformation to the concatination
        x = self.wo(x)

        # batch, seq len, d model
        return x

# taking the input from layer and saving it to add to the output.
class ResidualConnectionAndNorm(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        out =  self.dropout(self.norm(x + sublayer(x)))
        return out

# singular encoder block, self att -> residual -> ff -> residual
class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_and_norm_block = nn.ModuleList([ResidualConnectionAndNorm(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        # src mask is to mask the padded tokens, as we want to ignore them
        # lambda
        x = self.residual_connection_and_norm_block[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection_and_norm_block[1](x, lambda x: self.feed_forward_block(x))
        return x

# full encoder, stack of encoder blocks
class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # layers is modulelist of encoder blocks
        self.layers = layers

    def forward(self, x, src_mask):
        # for each layer, pass x and src to encoder block and do forward pass
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# decoder block, self attention + residual, cross attention + residual, ff + residual
class DecoderBlock(nn.Module):
    def __init__(self, features, masked_self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        # define attribtues as decoder sublayers needed
        self.masked_self_attention_block = masked_self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # 3 residual layers
        self.residual_connection_and_norm_block = nn.ModuleList([ResidualConnectionAndNorm(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # go into self atteniton, need only x,
        x = self.residual_connection_and_norm_block[0](x, lambda x: self.masked_self_attention_block(x, x, x, tgt_mask))
        # cross attention takes encoder output as kv and q from decoder output
        x = self.residual_connection_and_norm_block[1](x , lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # ff
        x = self.residual_connection_and_norm_block[2](x, lambda x: self.feed_forward_block(x))
        return x

# decoder blocks together
class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # layers of decoder blocks
        self.layers = layers

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # input params for each layer (decoder block)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

# final linear layer from output to output for all vocab (and softmax)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # turn output of decoder into vocab, which we will apply softmax to see probability of all vocab
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # no softmax 
        return self.linear(x)

# building the complete transformer
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
        # embdedding to pos to encoder to encoder to pos to decoder to linear 
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.encoder = encoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src is tokens
        src = self.src_embed(src)
        # now embeddings
        src = self.src_pos(src)
        # now embeddings with positional encoding
        src = self.encoder(src, src_mask)
        # now gone through the encoder, and calculated attention for all tokens in sentence (and in batch)
        # src_mask masks pads in src
        return src

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # input tgt is tokens (output so far)
        tgt = self.tgt_embed(tgt)
        # now embedding
        tgt = self.tgt_pos(tgt)
        # now embedding with positional encoding
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        # goes through decoder, returns as output rtepresentation in vector d_model(512)
        # src mask for masking pad again (in cross attnetion)
        # tgt mask for masking future (masked self attention)
        # encoder output for cross attention
        # tgt for self attention and then goes to cross attention + ff
        return tgt

    # final linear layer
    def projection(self, x):
        # linear layer to take output vecotr d_model, put into vector size vocab size to represent each word likeliness
        # in is output from decoder
        x = self.projection_layer(x)
        # out is final logits (or softmax)
        return x

# function : given all hyperparameters will build thee trasnformer with intial parameters
# creating the classes instances
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, n = 6, heads = 8, dropout = 0.1, d_ff = 2048):
    # create embedding layers and positional encoding
    # for input to encoder
    src_embed = InputEmbedding(d_model, src_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    # for output into decoder
    tgt_embed= InputEmbedding(d_model, tgt_vocab_size)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) 

    # create encoder blocks (n)
    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block =  EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # create decoder blocks (with self, cross, ff)
    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block =  DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # put list into decoder to create decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create linear projection 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # build transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialise the paramters, python does it for you
    return transformer