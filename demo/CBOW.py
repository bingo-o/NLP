import torch
from torch import optim, nn
import torch.nn.functional as F

context_size = 2

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)

word2id = {word: i for i, word in enumerate(vocab)}

dataset = []

for i in range(context_size, len(raw_text) - context_size):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    dataset.append((context, target))

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, dim=0, keepdim=True)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CBOW(len(word2id), 100, context_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(100):
    total_loss = 0
    for context, target in dataset:
        context = torch.LongTensor([word2id[word] for word in context]).to(device)
        target = torch.LongTensor([word2id[target]]).to(device)
        out = model(context)
        loss = criterion(out, target)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch[{} / {}], loss: {:.3f}".format(epoch + 1, 100, total_loss.item() / len(word2id)))