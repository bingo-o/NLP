import torch
from torch import nn
from const import *


class BilstmAtt(nn.Module):
    def __init__(self, args, embedding_pre):
        super(BilstmAtt, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.Tensor(embedding_pre), padding_idx=PAD, freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD)

        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=2 * LIMIT + 1)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=2 * LIMIT + 1)
        self.relation_embeds = nn.Embedding(self.label_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.pos_dim * 2, self.hidden_dim // 2, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.att_weight = nn.Parameter(torch.randn(self.batch_size, 1, self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch_size, self.label_size, 1))

    def init_hidden_lstm(self, device):
        nums = self.num_layers * 2
        return (torch.randn(nums, self.batch_size, self.hidden_dim // 2).to(device),
                torch.randn(nums, self.batch_size, self.hidden_dim // 2).to(device))

    def attention(self, H):
        M = torch.tanh(H)
        a = torch.softmax(torch.bmm(self.att_weight, M), dim=-1)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def forward(self, sents, pos1, pos2, device):
        self.hidden = self.init_hidden_lstm(device)
        embeds = torch.cat([self.word_embeds(sents), self.pos1_embeds(pos1), self.pos2_embeds(pos2)], dim=-1)
        lstm_out, _ = self.lstm(embeds, self.hidden)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = torch.tanh(self.attention(lstm_out))
        relation = torch.LongTensor(range(self.label_size)).repeat(self.batch_size, 1).to(device)
        relation = self.relation_embeds(relation)
        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)
        return res.view(self.batch_size, -1)
