import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(q.shape, k.shape, v.shape)

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

# Custom Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with Q, K, V extraction
        src2, attn_weights, q, k, v = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(src2))
        
        # Feedforward network with residual connection
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        
        return src, q, k, v  # Return Q, K, V

# Custom Transformer Decoder Layer
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with Q, K, V extraction
        tgt2, attn_weights, q, k, v = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        
        # Cross-attention with memory
        tgt2, attn_weights, q2, k2, v2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # Feedforward network with residual connection
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt, q, k, v  # Return Q, K, V
