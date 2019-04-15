import torch
from torch import optim, nn
import torch.nn.functional as F

num_epochs = 100
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_sentence = """n-gram models are widely used in statistical natural 
language processing . In speech recognition , phonemes and sequences of 
phonemes are modeled using a n-gram distribution . For parsing , words 
are modeled such that each n-gram is composed of n words . For language 
identification , sequences of characters / graphemes ( letters of the 
alphabet ) are modeled for different languages For sequences of characters , 
the 3-grams ( sometimes referred to as " trigrams " ) that can be 
generated from " good morning " are " goo " , " ood " , " od " , " dm ", 
" mo " , " mor " and so forth , counting the space character as a gram 
( sometimes the beginning and end of a text are modeled explicitly , adding 
" __g " , " _go " , " ng_ " , and " g__ " ) . For sequences of words , 
the trigrams that can be generated from " the dog smelled like a skunk " 
are " # the dog " , " the dog smelled " , " dog smelled like ", " smelled 
like a " , " like a skunk " and " a skunk # " .""".split()

# 构造数据，分别为训练数据和目标结果
trigrams = [([test_sentence[i], test_sentence[i+1]],
            test_sentence[i+2]) for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)

# word to index
word2id = {word: i for i, word in enumerate(vocab)}

# index to word
id2word = {i: word for word, i in word2id.items()}

def context2ids(context):
    return list(map(lambda w: word2id[w], context))

class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_size=16, context_size=2):
        # 初始化父类
        super(NGram, self).__init__()
        # 定义embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 两层线性回归，多一层参数，提高泛化性
        self.fc1 = nn.Linear(context_size * embedding_size, 128)
        # vocab_size输出类别数，这里即为单词数
        self.fc2 = nn.Linear(128, vocab_size)
    
    def forward(self, x):
        """
        向前传播
        """
        embeds = self.embedding(x).view(1, -1)
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

class Model:
    def __init__(self, num_epochs=num_epochs, learning_rate=learning_rate):
        self.model = NGram(len(vocab)).to(device) # 初始化模型
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
    def train(self, train_dataset):
        criterion = nn.NLLLoss() # 代价函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # 定义优化函数
        for epoch in range(self.num_epochs):
            total_loss = 0
            for context, target in train_dataset:
                context = torch.LongTensor(context2ids(context)).to(device)
                target = torch.LongTensor([word2id[target]]).to(device)
                out = self.model(context)
                loss = criterion(out, target)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch[{} / {}], loss: {:.3f}".format(epoch + 1, self.num_epochs, total_loss.item() / len(train_dataset)))
            
    def predict(self, context):
        context = torch.LongTensor(context2ids(context))
        out = self.model(context)
        index = torch.argmax(out, dim=1).item()
        return id2word[index]

if __name__ == "__main__":
    model = Model()
    model.train(trigrams)
    for context in [["widely", "used"], ["and", "so"], ["are", "modeled"]]:
        print("{} + {} = {}".format(context[0], context[1], model.predict(context)))