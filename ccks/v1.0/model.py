import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Bilstm(nn.Module):
    def __init__(self, args, embedding_pre):
        super(Bilstm, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.Tensor(embedding_pre), padding_idx=0, freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        self.pos_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.lstm = nn.LSTM(self.embedding_dim + self.pos_dim, self.hidden_dim // 2, num_layers=self.num_layers,
                            batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.label_size)

    def init_hidden_lstm(self, device):
        nums = 2 * self.num_layers
        return (torch.randn(nums, self.batch_size, self.hidden_dim // 2).to(device),
                torch.randn(nums, self.batch_size, self.hidden_dim // 2).to(device))

    def forward(self, sentence, pos, length, device):
        self.hidden = self.init_hidden_lstm(device)
        embeds = torch.cat([self.word_embeds(sentence), self.pos_embeds(pos)], dim=-1)
        packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
        lstm_out, _ = self.lstm(packed_embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = torch.Tensor().to(device)
        for i in range(len(length)):
            out = torch.cat([out, lstm_out[i][length[i] - 1].unsqueeze(0)], dim=0)
        return self.fc(out)
