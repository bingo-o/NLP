import torch
from torch import nn
from torch.autograd import Variable
from const import *


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)


class LSTM_Text(nn.Module):

    def __init__(self, args):
        super(LSTM_Text, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.num_directions = 2 if self.bidirectional else 1
        self.lookup_table = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=PAD)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                            dropout=self.dropout, bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size * self.num_directions)
        self.fc = nn.Linear(
            self.hidden_size * self.num_directions, self.label_size)
        self.init_weight()

    def init_weight(self, scope=0.1):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.fc.weight.data.uniform_(-scope, scope)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.lookup_table(x)
        out, _ = self.lstm(x.transpose(0, 1))
        out = self.ln(out)[-1]
        return self.fc(out)
