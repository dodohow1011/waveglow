import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class waveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(waveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)

    def log_alignment(self, model, enc_slf_attn, dec_enc_attn, out_mel, target, iteration):

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        #idx = random.randint(0, enc_slf_attn.size(0) - 1)
        idx = 0
        self.add_image(
            "encoder_self_alignment",
            plot_alignment_to_numpy(enc_slf_attn[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "decoder_encoder_alignment",
            plot_alignment_to_numpy(dec_enc_attn[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(target[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(out_mel[idx].data.cpu().numpy()),
            iteration)
