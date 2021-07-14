import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import make_std_mask
from module import process_data
from utils import UNK, BOS, PAD

__all__ = ['GreedyGenerator']


class GreedyGenerator(nn.Module):
    def __init__(self, model, max_tgt_len, multi_gpu=False):
        super(GreedyGenerator, self).__init__()
        if multi_gpu:
            self.model = model.module
        else:
            self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = BOS
        self.unk_pos = UNK

    def forward(self, data):
        process_data(data)
        src_seq = data.src_seq
        src_mask = src_seq.eq(PAD)
        src_emb = self.model.src_embedding(src_seq)
        data.src_emb = src_emb
        encoder_outputs = self.model.encode(data)

        batch_size = src_seq.size(0)
        ys = torch.ones(batch_size, 1).fill_(self.start_pos).type_as(src_seq.data)
        for i in range(self.max_tgt_len - 1):
            tgt_mask = make_std_mask(ys, 0)

            tgt_emb = self.model.tgt_embedding(Variable(ys))
            decoder_outputs, decoder_attn = self.model.decode(tgt_emb, encoder_outputs, tgt_mask, src_mask)
            out = self.model.generator(decoder_outputs)
            out = out[:, -1, :]
            _, next_word = torch.max(out, dim=1)
            ys = torch.cat([ys,
                            next_word.unsqueeze(1).type_as(src_seq.data)], dim=1)

        return ys[:, 1:]

