import torch
from torch import nn
import torch.nn.functional as F
from const import *


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=PAD)
        self.encoders = nn.ModuleList([nn.Conv2d(
            1, self.num_kernels, (filter_size, self.embed_size)) for filter_size in self.filter_sizes])
        self.fc = nn.Linear(
            len(self.filter_sizes) * self.num_kernels, self.label_size)
        self.dropout = nn.Dropout(self.dropout)
        self.init_weight()

    def forward(self, x):
        n_id = 0
        c_id = 1
        h_id = 2
        w_id = 3

        x = self.lookup_table(x)
        x = x.unsqueeze(c_id)

        enc_outs = []
        for encoder in self.encoders:
            enc_out = F.relu(encoder(x), inplace=True)
            height = enc_out.size(h_id)
            enc_out = F.max_pool2d(enc_out, (height, 1))
            enc_out = enc_out.squeeze(w_id)
            enc_out = enc_out.squeeze(h_id)
            enc_outs.append(enc_out)

        inputs = self.dropout(torch.cat(enc_outs, dim=1))
        return self.fc(inputs)

    def init_weight(self, scope=0.1):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.fc.weight.data.uniform_(-scope, scope)
