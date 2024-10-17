# ====================================================================================================================================================================
#                                                            PYTORCH TRANSFORMER IMPLEMENTATION
# ====================================================================================================================================================================
# Research Papaer - Attention is all you need: https://arxiv.org/abs/1706.03762
# Referred to tutorial: https://www.youtube.com/watch?v=ISNdQcPhsts

# Imports:
import torch
import torch.nn as nn
import math

# ====================================================================================================================================================================
#                                                                       Sub Layers:
# ====================================================================================================================================================================
# Class for Layer Normalization: Ensures stable and accelerated training after each sub-layer (self-attention or feed forward)
# Z-score Normalized Output = alpha * (x - mean) / (std + ε) + bias
class LayerNormalization(nn.Module): 

    # Constructor:
    # Features refers to the size of each token's embedding vector: d_model
    # We compute the last dimension (i.e., d_model), as it computes the mean and variance for each token's feature values independently.
    def __init__(self, features: int, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps                                      # Epsilon is present for numerical stability (in the denominator to ensure the value doesn't get too big)
        self.alpha = nn.Parameter(torch.ones(features))     # Learnable parameters - Multiplied (rescale the normalized output)
        self.bias = nn.Parameter(torch.ones(features))      # Learnable parameters - Added (shift the normalize output)

    # Calculate the mean and standard deviations and return the normalization. 
    def forward(self, x):
        # Referring to the last dimension (-1) of the 3D tensor
        # Additionally when calculating mean/std, they tend to reduce the dimensions by default hence we keep it true to retain dimension shape
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        # Return the z-score normalization 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# This is a Feed Forward Network Layer in the Encoder which has two linear layers with a ReLu activation in between.
# This sub layer is responsible for capturing more detailed patterns and features from the outputted Attention sub layer. 
# FFN(x) = max(0, x * w1 + b1)w2 + b2
class FeedForwardBlock(nn.Module): 

    # Constructor:
    # From the paper we input the d_model (size of embedding vector)
    # 'dff' - refers to the hidden layer dimensionality inside the feed-forward network
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    # We first compute the input through Linear 1 (projects d_model dimensionality to dff) then after applying ReLu with Linear 2 (projects dff back to d_model dimensionality)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Class for Word Embeddings: Converting text into numerical vectors of d_model (512)
class InputEmbeddings(nn.Module): 

    # Constructor:
    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__()
        self.d_model = d_model                                  # Represents the model dimensionality - the size of each vector for each word
        self.vocab_size = vocab_size                            # Represents the total vocabulary within the dataset
        self.embedding = nn.Embedding(vocab_size, d_model)      # Create the embedding layer function to which we can use for any input sequence of words

    # Based on the research paper - Multiply by sqrt(d_model) to scale the embeddings
    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.d_model)

# Class for Positional Embeddings: Tracking the position of words relative to the sentence
class PositionalEncoding(nn.Module): 

    # Constructor:
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model                  # Embedding vector size for each word
        self.seq_len = seq_len                  # Represents the maximum length of the input sequence
        self.dropout = nn.Dropout(dropout)      # Regularization for positional encoding to ensure that the model doesn't overfit - ex: dropout - 0.1, means that 10% of the elements in the input tensor will be 0

        # Create a matrix of shape (seq_len, d_model) - 2D matrix representing each word and their d_model vector
        pe = torch.zeros(seq_len, d_model)

        # Creates a vector of shape (seq_len,) filled with values from 0 to seq_len - 1
        # Position represents the position that word is in the sequence of the sentence.
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # unsequeeze makes it shape as (seq_len, 1)

        # Creates division term vector - basically mathematically equates to: 1/(10000 ** (2i/d_model)) - where i is the ith index of the d_model vector per word
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even words positions
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd words positions
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        # Add another batch dimension in the beginning [0] - This is to allow for parallel processing of multiple batches of positional encoded vectors
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # This is to ensure the positional encoded data is saved as a buffer within the module - it will not be a learnable parameter.
        # Essentially we ensure that it is included when saving/loading the model’s state.
        self.register_buffer("pe", pe)

    # We add the positional encoding on top of the word embeddings (every word within the sentence)
    # We do all batches, till the end of the input sequence x.shape[1] and for all embedding dimensions (d_model)
    def forward(self, x): 
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False) # Prevents gradients from being computed for the PE matrix when performing backpropagation as they are not learnable. 
        return self.dropout(x)

# Class for Residual Connections: Providing the pre-inputs into the normalization layer that proceeds a particular sublayer (attention or feed forward)
# These skip connections are used to maintain better flow of gradients through the network during backpropogation and help to preserve original information if lost. 
class ResidualConnection(nn.Module):

    # Constructor: 
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    # Add the pre-input embeddings into the normalized sublayer embeddings
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# This takes the embeddings as an input 3 times: One for queries, keys and values of shape (seq_len, d_model) for each
# - Queries: Used for determining the relevance of its word with other key vectors of all other words in the sequence (including itself)
# - Keys: Used for comparing its word with the query to determine the similarity/relevance for each word in the sequence.
# - Values: Represents the actual content/information of each token that will be used to produce the final output of the attention mechanism.

# We then multiply these matrices with weighted linear layers to get new matricies: Q', K', V'
# Then we split these results into several 'heads' - h matricies along the embedding dimension (in order to get access to the full sentence)
# We apply attention functionality to each of these heads and then we combine them along with another linear layer to result in the multi-head attention output.

# MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * Wo, where head_i = Attention(Q * Wq_i, K * Wk_i, V * Wv_i)
class MultiHeadAttentionBlock(nn.Module): 

    # Constructor:
    def __init__(self, d_model: int, h: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model                  # Embedding vector size
        self.h = h                              # Number of heads
        self.dropout = nn.Dropout(dropout)

        # We ensure that d_model is divisible by h - whole output
        assert d_model % h == 0, "d_model is not divisible by h"

        # d_k: Dimension of vector seen by each head (dimension shape of keys/queries)
        self.d_k = d_model // h              

        # We setup the linear weighted layers of shape (d_model, d_model) which are multiplied with the queries, keys and values to produce new matricies: Q', K', V'
        # These layers help us learn distinct representations for each of these components (Q, K, V).They help learn how to transform the original input embeddings to cater the role of queries, keys, values.
        # For example: Wq learns how to transform the original input embeddings to create queries that capture specific relationships or dependencies.
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

        # Weighted Linear Layer for the concatonation of all the heads to provide the final Multi-head attention output
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    # Scaled Dot-Product Attention Functionality
    # Attention = softmax((Q * K)/d_k) * V
    @staticmethod # Can call this function without having an instance of the class 
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Get the attention scores via matrix multiplication (@) between the query and key (tranposed this to shape (d_k, seq_len)) and then scaled
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Applying a mask where we write a very low value (indicating -inf) to the positions where mask == 0, so softmax will declare them as 0
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Compute softmax
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) 

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention scores multiplied with values as the output and the attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    # Overall Multi-head attention functionality:
    def forward(self, q, k, v, mask): 
        # Pass the Queries, Keys and Values through weighted linear layers
        query = self.w_q(q)         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)           # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)         # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  

        # Here we split the outputted matricies (Q', V', K') embeddings into 'h' heads
        # We keep the batch, the seq_len and we are splitting the embedding dimensions into d_k (dimension size of head) along h number of heads
        # Then we transpose tensor so that the heads 'h' is second because each head has a shape of (d_k, seq_len)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Now we calculate the attention resulting in the output and the attention scores (output of the softmax)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Now we must combine all the heads together (concatonation)
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by the final Linear Layer Wo and return the output as the final multi-head attention output
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

# This is the final Linear layer which projects the outputs into the final probabilities
class ProjectionLayer(nn.Module):

    # Constructor for setting up the linear layer
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

# ====================================================================================================================================================================
#                                                                Encoder and Decoder Blocks:
# ====================================================================================================================================================================
# This represents a single Encoder Block of the encoder which comprises of the sublayers shown on the paper (Self-attention, Feedforward, 2 Normalization and 2 Residuals)
class EncoderBlock(nn.Module): 
    
    # Constructor: 
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block    # Multi-head self attention sub layer
        self.feed_forward_block = feed_forward_block        # Feed forward sub layer
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])    # Two residual connections (one for each sub layer)

    # Takes the input tensor to the encoder block and a source mask (used to control which tokens in the input sequence should be attended to during the self-attention operation)
    def forward(self, x, src_mask):
        # First residual connection between the positional embedding input and the other input from the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # Second residual connection between the input after the first normalize layer and the other input from the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# This class repersents the entire Encoder stack in the transformer which can consist of (Nx) EncoderBlock instances stacked together
class Encoder(nn.Module):

    # Constructor for setting up many EncoderBlock instances
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers    # List of EncoderBlock instances          
        self.norm = LayerNormalization(features)

    # Iterates through each EncoderBlock and updates the input tensor (x) by passing it through each layer.
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# This represents a single Decoder Block of the decoder which comprises of the sublayers shown on the paper (2 Self-attention, Feed Forward, 3 Normalization and 3 Residuals)
class DecoderBlock(nn.Module):

    # Constructor: 
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block        # First attention block in decoder
        self.cross_attention_block = cross_attention_block      # Second attention block in decoder (one that connects to encoder)
        self.feed_forward_block = feed_forward_block            # Feed forward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    # We give the second self-attention block different inputs since this one retrieves another input from the encoder's output
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# This class repersents the entire Decoder stack in the transformer which can consist of (Nx) DecoderBlock instances stacked together
class Decoder(nn.Module):

    # Constructor for setting up many DecoderBlock instances
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    # Iterates through each DecoderBlock and updates the input tensor (x) by passing it through each layer.
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# ====================================================================================================================================================================
#                                                                       Transformer:
# ====================================================================================================================================================================
# This class represents the encoding functionality, decoding functionality and final linear layer projection.
class Transformer(nn.Module):

    # Constructor: defining all the different sub layers
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer,) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Here we apply the encoding
    def encode(self, src, src_mask):
        # Apply the word embedding, positional encoding and then the overall encoder
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    # Here we apply the decoding
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor,):
        # Apply the word embedding, positional encoding and then the overall encoder
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # Final linear layer projection
    def project(self, x):
        return self.projection_layer(x)


# ====================================================================================================================================================================
#                                                                 Final Building Function:
# ====================================================================================================================================================================
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048,) -> Transformer: 
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters so they don't start with some random values (improves training)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
