import torch
from torch import nn
import torch.nn.functional as F

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_lang_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_lang_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)
        out, hidden = self.gru(x, hidden)
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
# 解码器
class Decoder(nn.Module):
    def __init__(self, output_lang_size, hidden_size, dropout, max_len):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_lang_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_len)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_lang_size)
        
    def forward(self, x, hidden, encoder_outs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)
        attn_weights = F.softmax(self.attn(torch.cat([x[0], hidden[0]], dim=1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outs.unsqueeze(0))
        x = self.attn_combine(torch.cat([x[0], attn_applied[0]], dim=1)).unsqueeze(0)
        x = F.relu(x)
        out, hidden = self.gru(x, hidden)
        out = self.out(out[0])
        return out, hidden, attn_weights
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)