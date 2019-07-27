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
import argparse
import json
import os
import sys
import torch

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
# from mel2samp import Mel2Samp
from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from utils import to_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(hparams).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def parse_batch(batch):
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return text_padded, input_lengths, mel_padded, max_len, output_lengths

def train(num_gpus, rank, group_name, output_directory, checkpoint_path, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(hparams.sigma)
    model = WaveGlow(hparams).cuda()

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path:
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = TextMelLoader(hparams.training_files, hparams)
    collate_fn = TextMelCollate()
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    batch_size = hparams.batch_size
    train_loader = DataLoader(trainset, num_workers=0, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    # Get shared output_directory readya

    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if hparams.with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    print ("Total Epochs: {}".format(hparams.epochs))
    print ("Batch Size: {}".format(hparams.batch_size))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            text_padded, input_lengths, mel_padded, max_len, output_lengths = parse_batch(batch)
            # mel_padded = mel_padded.transpose(1, 2)
            src_pos = torch.arange(hparams.n_position)
            src_pos = to_gpu(src_pos).long().unsqueeze(0)
            src_pos = src_pos.expand(hparams.batch_size, -1)
            '''print (src_pos.size())
            print (input_lengths)
            print (max_len)
            print (output_lengths)
            print (hparams.n_symbols)
            print (src_pos)
            sys.exit()'''
            outputs = model(mel_padded, text_padded, src_pos)

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))
            if hparams.with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if (iteration % hparams.iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    '''parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')'''
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    num_gpus = 1
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
    train(num_gpus, args.rank, args.group_name, args.output_directory, args.checkpoint_path, hparams)
