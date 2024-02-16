#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import sys
sys.path.insert(0, '/audio_transformer')

from multiprocessing import freeze_support
import platform
from tqdm import tqdm
from data_handling.audiocaps_dataset import get_audiocaps_loader
from tools.config_loader import get_config
from tools.utils import *
from tools.file_io import load_pickle_file
from models.TransModel import ACT
from tools.beam import beam_decode
import warnings
from models.SentenceModel import SentenceModel
import math
from sentence_attention import SentenceAttention
import json
import torch

warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning) 


class Audio2Caption():
    def __init__(self, config_path):
        config = get_config(config_path)
        setup_seed(config.training.seed)

        self.device, device_name = (torch.device('cuda'),
                               torch.cuda.get_device_name(torch.cuda.current_device())) \
            if torch.cuda.is_available() else ('cpu', platform.processor())

        self.words_list = load_pickle_file(config.path.vocabulary)
        self.sos_ind = self.words_list.index('<sos>')
        self.eos_ind = self.words_list.index('<eos>')

        self.test_data = get_audiocaps_loader('test', config)

        ntokens = len(self.words_list)

        self.model = ACT(config, ntokens)
        self.model.load_state_dict(torch.load(config.path.eval_model)['model'])
        self.model.to(self.device)
        self.model.eval()
        self.sa = SentenceAttention()

    def data_preprocessing(self):
        pass

    def positional_encoding(self, x, data, data_len):
        y = 1/(2 + np.exp(2-0.5*x)) + data
        return y
    
    def eval_greedy(self, data):
        with torch.no_grad():
            y_hat_all = []
            ref_captions_dict = []
            file_names_all = []
            y_hat_pb_all = []

            for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):
                src, target_dicts, file_names = eval_batch
                src = src.to(self.device)
                output, output_pb = greedy_decode(self.model, src, sos_ind=self.sos_ind, eos_ind=self.eos_ind)

                output = output[:, 1:].int()
                output_pb = output_pb[:,1:].float()

                y_hat_batch = torch.zeros(output.shape).fill_(self.eos_ind).to(self.device)
                y_hat_pb_batch = torch.zeros(output_pb.shape).to(self.device)

                for i in range(output.shape[0]):    # batch_size
                    for j in range(output.shape[1]):
                        y_hat_batch[i, j] = output[i, j]
                        y_hat_pb_batch[i, j ] = output_pb[i, j]
                        if output[i, j] == self.eos_ind:
                            break
                        elif j == output.shape[1] - 1:
                            y_hat_batch[i, j] = self.eos_ind
                            y_hat_pb_batch[i, j] = 9999

                y_hat_batch = y_hat_batch.int()
                y_hat_pb_batch = y_hat_pb_batch.float()

                y_hat_all.extend(y_hat_batch.cpu())
                y_hat_pb_all.extend(y_hat_pb_batch.cpu())

                ref_captions_dict.extend(target_dicts)
                file_names_all.extend(file_names)

            captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict,
                                                           file_names_all, self.words_list)

            captions_attention = []
            result_captions = []
            for p, c in zip(y_hat_pb_all, captions_pred):
                p = p[:len(c['caption_predicted'].split(' '))]
                captions_attention.append([round(item.cpu().item(),4) for item in p])
                result_captions.append(c['caption_predicted'])

            return result_captions, captions_attention, file_names_all

    def audio2txt(self):
        captions_pred, captions_a, file_names_all = self.eval_greedy(self.test_data)
        re_captions_a = []

        for c_idx, caption_a in enumerate(captions_a):
            sa_result = self.sa.get_attention(captions_pred[c_idx])
            for idx, item in enumerate(sa_result):
                if item > 0.7:
                    caption_a[idx] = float(item) + 0.1*caption_a[idx]
                else:
                    caption_a[idx] = caption_a[idx]

            r_caption_attention = []
            for w_idx, word_a in enumerate(caption_a):
                positional_encoding_result = self.positional_encoding(w_idx, word_a, len(caption_a))
                r_caption_attention.append(positional_encoding_result)

            re_captions_a.append(r_caption_attention)

        return captions_pred, re_captions_a, file_names_all
