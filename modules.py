import sys
import torch
import torch.nn as nn
import numpy as np
from EncLayer import EncoderLayer

def position_encoding(n_position, d_hid, padding_idx=None):
    
    def cal_angle(position, hid_idx):
       return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, hparams)

        super(Encoder, self).__init__()

        self.d_model = hparams.d_model
        self.n_position = hparams.n_position
        self.n_src_vocab = hparams.n_symbols
        self.d_word_vec = hparams.symbols_embedding_dim
        self.n_head = hparams.n_head
        self.d_inner = hparams.d_hidden
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.n_layers = hparams.n_layers
        self.dropout = hparams.dropout

        self.src_word_emb = nn.Embedding(self.n_src_vocab, self.d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.n_position, self.d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=self.dropout)
            for _ in range(self.n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
