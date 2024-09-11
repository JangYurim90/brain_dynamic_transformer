import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Weight matrices for query, key, value
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Ensure the dimensions match
        assert query.size(-1) == self.embed_dim, f"Query embed_dim {query.size(-1)} doesn't match {self.embed_dim}"
        assert key.size(-1) == self.embed_dim, f"Key embed_dim {key.size(-1)} doesn't match {self.embed_dim}"
        assert value.size(-1) == self.embed_dim, f"Value embed_dim {value.size(-1)} doesn't match {self.embed_dim}"

        # Compute Q, K, V manually using linear projection
        q, k, v = self._in_proj_qkv(query, key, value)

        # Call MultiheadAttention with manually computed Q, K, V
        attn_output, attn_weights = self.multihead_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Return attention output, attention weights, and Q, K, V
        return attn_output, attn_weights, q, k, v
    
    def _in_proj_qkv(self, query, key, value):
        """Manually compute the Q, K, V matrices for attention."""
        # Weight slicing
        q_proj_weight = self.in_proj_weight[:self.embed_dim, :]
        k_proj_weight = self.in_proj_weight[self.embed_dim:2 * self.embed_dim, :]
        v_proj_weight = self.in_proj_weight[2 * self.embed_dim:, :]

        # Linear projections
        q = F.linear(query, q_proj_weight, self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, k_proj_weight, self.in_proj_bias[self.embed_dim:2 * self.embed_dim])
        v = F.linear(value, v_proj_weight, self.in_proj_bias[2 * self.embed_dim:])

        return q, k, v
    
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, activation='gelu'):
        super(CustomTransformerEncoderLayer, self).__init__()

        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Use CustomMultiheadAttention to get Q, K, V and output
        attn_output, attn_weights, Q, K, V = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        # The rest of the transformer encoder layer
        src2 = self.dropout1(attn_output)
        src = src + self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.norm2(src2)
        
        return src, Q, K, V


    
class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, output_length=10, dropout=0.1,
                 activation='relu', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.output_length = output_length  # 추가: 출력 타임포인트 개수

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        encoder_layer = CustomTransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, output_length, feat_dim) -> 다음 10개 타임포인트 예측
            Q, K, V: extracted query, key, and value from attention
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model]
        inp = self.pos_enc(inp)

        # Prepare to store Q, K, V from all layers
        all_Q, all_K, all_V = [], [], []

        # Pass through transformer encoder layers
        for layer in self.transformer_encoder:
            inp, Q, K, V = layer(inp, src_key_padding_mask=~padding_masks)
            all_Q.append(Q)
            all_K.append(K)
            all_V.append(V)

        output = self.act(inp)  # (seq_length, batch_size, d_model)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        # 마지막 10개의 타임포인트만 반환
        output = output[:, -self.output_length:, :]  # (batch_size, output_length, feat_dim)

        return output, all_Q, all_K, all_V

    
