# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from layers import ConvNorm
from Encoder import Encoder
from Attention import ScaledDotProductAttention
from SubLayer import DecoderLayer
import torch.nn.functional as F
import sys


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def is_end_of_frames(mel, eps=0.2):
    return (output.data <= eps).all()



class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W



class Decoder(nn.Module):
    def __init__(self, d_o, d_inner, n_head, d_mel_half, d_k, d_v, dropout=0.1):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(d_mel_half, d_o)  # 40 -> 256
        self.decoder = DecoderLayer(d_o, d_inner, n_head, d_k, d_v, dropout=dropout)
    
    def forward(self, mel_0, enc_output):
        
        mel_0 = self.linear(mel_0) # 40 -> 256
        dec_output, dec_enc_attn = self.decoder(mel_0, enc_output)
        return dec_output, dec_enc_attn

class WaveGlow(nn.Module):
    def __init__(self, hparams):
        super(WaveGlow, self).__init__()

        self.upsample = nn.ConvTranspose1d(hparams.n_mel_channels,
                                           hparams.n_mel_channels,
                                           1024, stride=256)
        assert(hparams.n_group % 2 == 0)
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size

        # model parameters
        self.d_model = hparams.d_model                        # 512
        self.d_inner = hparams.d_hidden
        self.n_position = hparams.n_position                  # 200
        self.n_symbols = hparams.n_symbols                    
        self.embedding_dim = hparams.symbols_embedding_dim    # 512
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k                                # 64
        self.d_v = hparams.d_v                                # 64
        self.n_layers = hparams.n_layers
        self.dropout = hparams.dropout
        self.d_o = hparams.d_o                                # 256
        self.max_mel_steps = 1000
        
        self.decoder = torch.nn.ModuleList()
        self.convinv = nn.ModuleList()
        self.encoder = Encoder(self.d_model, self.n_position, self.n_symbols, self.embedding_dim, self.n_head, self.d_k, self.d_v, self.n_layers, self.dropout)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = hparams.n_mel_channels
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.decoder.append(Decoder(d_o=self.d_o, d_inner=self.d_inner, d_mel_half =n_remaining_channels//2, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, dropout=self.dropout))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, mel, words, src_pos):

        # mel: B x D x T
        # words: B x T

        output_mel = []
        log_s_list = []
        log_det_W_list = []

        enc_output = self.encoder(words, src_pos)

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_mel.append(mel[:,:self.n_early_size,:])
                mel = mel[:,self.n_early_size:,:]

            mel, log_det_W = self.convinv[k](mel)
            log_det_W_list.append(log_det_W)

            n_half = int(mel.size(1)/2)
            mel_0 = mel[:,:n_half,:]
            mel_1 = mel[:,n_half:,:]


            output, dec_enc_attn = self.decoder[k]((mel_0, enc_output))
            log_s = output[:, n_half:, :]
            t = output[:, :n_half, :]
            mel_1 = torch.exp(log_s)*mel_1 + t
            log_s_list.append(log_s)

            mel = torch.cat([mel_0, mel_1],1)

        output_mel.append(mel)
        return torch.cat(output_mel,1), log_s_list, log_det_W_list

    def infer(self, words, sigma=1.0):

        enc_output = Encoder(words) # b * d_enc * t

        mel = torch.cuda.FloatTensor(enc_output.size(0), self.n_remaining_channels, self.max_mel_steps).normal_()
        mel = torch.autograd.Variable(sigma*mel)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            mel_0 = mel[:,:n_half,:]
            mel_1 = mel[:,n_half:,:]

            output = self.TF[k]((mel_0, enc_output))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            mel_1 = (mel_1 - b)/torch.exp(s)
            mel = torch.cat([mel_0, mel_1],1)

            mel = self.convinv[k](mel, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.cuda.FloatTensor(enc_output.size(0), self.n_early_size, max_mel_steps).normal_()
                mel = torch.cat((sigma*z, mel),1)
        
        for t in mel.size(2):
            if t > 1 and is_end_of_frames(mel[:, :, t]):
                mel = mel[:, :, :t+1]   # get the end of mel

        mel = mel.permute(0,2,1).data # b * t * d_mel
        return mel

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = nn.ModuleList()
    for old_conv in conv_list:
        old_conv = nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
