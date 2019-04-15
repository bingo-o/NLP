import torch
from torch import optim, nn
import torch.nn.functional as F
import string

num_epochs = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = [("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

word2id = {}
tag2id = {}
id2tag = {}

for context, tag in training_data:
    for word in context:
        if word not in word2id:
            word2id[word] = len(word2id)

    for label in tag:
        if label not in tag2id:
            tag2id[label] = len(tag2id)
            id2tag[len(id2tag)] = label

alphabet = string.ascii_lowercase
character2id = {}
for i in range(len(alphabet)):
    character2id[alphabet[i]] = i


def make_sequence(x, dic):
    return torch.LongTensor([dic[i] for i in x])


class CharLSTM(nn.Module):

    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)
        out, _ = self.char_lstm(x)
        return out[:, -1, :]


class LSTMTagger(nn.Module):

    def __init__(self, n_word, n_dim, n_hidden, n_char, char_dim, char_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_tag)

    def forward(self, x, word):
        char = torch.FloatTensor().to(device)
        for each in word:
            char_list = []
            for letter in each:
                char_list.append(character2id[letter.lower()])
            char_list = torch.LongTensor(char_list).unsqueeze(0).to(device)
            temp_char = self.char_lstm(char_list)
            char = torch.cat([char, temp_char], dim=0)
        x = self.word_embedding(x)
        x = torch.cat([x, char], dim=1)
        x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        out = out.squeeze(0)
        out = self.fc(out)
        return out


model = LSTMTagger(len(word2id), 100, 128, len(
    character2id), 10, 50, len(tag2id)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(word, word2id).to(device)
        tag = make_sequence(tag, tag2id).to(device)
        out = model(word_list, word)
        loss = criterion(out, tag)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{} / {}], loss: {:.3f}".format(epoch + 1,
                                                num_epochs, total_loss.item() / len(training_data)))

inputs = make_sequence("Everybody ate the apple".split(), word2id).to(device)
out = model(inputs, "Everybody ate the apple".split())
tag = [id2tag[each.item()] for each in torch.argmax(out, dim=1)]
print(tag)