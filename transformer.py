import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from custom_layer import CustomMultiheadAttention, CustomTransformerEncoderLayer, CustomTransformerDecoderLayer


# Positional Encoding class (unchanged)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Main Transformer model using custom encoder and decoder layers
class TFModel(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers, input_length, output_length, dropout=0.5):
        super(TFModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder with custom MultiheadAttention
        encoder_layers = CustomTransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.ModuleList([encoder_layers for _ in range(nlayers)])
        
        # Decoder with custom MultiheadAttention
        decoder_layers = CustomTransformerDecoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_decoder = nn.ModuleList([decoder_layers for _ in range(nlayers)])

        self.d_model = d_model
        self.fc_out = nn.Linear(d_model, output_length)  # Predict a single value per timepoint and ROI

        self.input_length = input_length
        self.output_length = output_length

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encoder
        src = self.pos_encoder(src)
        memory, enc_qkv = self.transformer_encoder_with_qkv(src, src_mask)  # Get Q, K, V from encoder

        # Decoder
        tgt = self.pos_encoder(tgt)
    

        output, dec_qkv = self.transformer_decoder_with_qkv(tgt, memory, tgt_mask, src_mask)  # Get Q, K, V from decoder

        output = self.fc_out(output)  # Convert to (batch_size, output_length, d_model)

        return output, enc_qkv, dec_qkv  # Return Q, K, V from both encoder and decoder

    def transformer_encoder_with_qkv(self, src, src_mask):
        qkv_values = []
        for layer in self.transformer_encoder:
            memory, q, k, v = layer(src, src_mask)
            qkv_values.append((q, k, v))

        return memory, qkv_values  # Return memory and QKV for encoder

    def transformer_decoder_with_qkv(self, tgt, memory, tgt_mask, src_mask):
        qkv_values = []
        for layer in self.transformer_decoder:
            output, q, k, v = layer(tgt, memory, tgt_mask, src_mask)
            qkv_values.append((q, k, v))

        return output, qkv_values  # Return output and QKV for decoder


    def generate_square_subsequent_mask_3(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def generate_square_subsequent_mask(self, tgt_len, src_len):
        """
        Generates an upper triangular mask of -inf and 0s, 
        to prevent the model from attending to future positions in the target sequence.
        
        Args:
            tgt_len (int): The length of the target sequence.
            src_len (int): The length of the source sequence.
        
        Returns:
            torch.Tensor: A mask of size (tgt_len, src_len) with 0s on the diagonal and -inf above the diagonal.
        """
        mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=1)  # 상삼각 행렬 생성
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 1인 부분을 -inf로 변환
        return mask  # (tgt_len, src_len)의 마스크 반환

    def generate_square_subsequent_mask_2(self,sz):
        """
        Generates an upper triangular mask of -inf and 0s, 
        to prevent the model from attending to future positions in the target sequence.
        
        Args:
            sz (int): The size of the mask (sequence length).
        
        Returns:
            torch.Tensor: A mask of size (sz, sz) with 0s on the diagonal and -inf above the diagonal.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # 상삼각 행렬 생성
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 1인 부분을 -inf로 변환
        return mask
