import argparse
import torch
from torch import optim, nn
from data_loader import DataLoader
from model import Bilstm
from tqdm import tqdm

parser = argparse.ArgumentParser(description="people relation extraction")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate [default: 1e-3]")
parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs for train")
parser.add_argument("--batch-size", type=int, default=128, help="batch size for training [default: 128]")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--save-best-model", type=str, default="./best_model.pt", help="path to save the best model")
parser.add_argument("--save-final-model", type=str, default="./final_model.pt", help="path to save the final model")
parser.add_argument("--data", type=str, default="./preprocessing/corpus.pt", help="location of the data corpus")
parser.add_argument("--embedding-dim", type=int, default=300, help="embedding dim [default: 300]")
parser.add_argument("--pos-dim", type=int, default=25, help="pos dim [default: 25]")
parser.add_argument("--hidden-dim", type=int, default=200, help="lstm hidden dim [default: 200]")
parser.add_argument("--num-layers", type=int, default=3, help="biLSTM layer numbers")
parser.add_argument("--cuda-able", action="store_true", help="enables cuda")
parser.add_argument("--pretrained", action="store_true", help="use pretrained embedding")

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
num_devices = torch.cuda.device_count()
device = torch.device("cuda:" + str(num_devices - 1) if use_cuda else "cpu")

data = torch.load(args.data)
args.vocab_size = data["dict"]["vocab_size"]
args.label_size = data["dict"]["label_size"]
args.pos_size = 3
word2id = data["dict"]["vocab"]
train_dataloader = DataLoader(data["train"]["sents"], data["train"]["labels"], data["train"]["indexes"],
                              batch_size=args.batch_size, shuffle=False)
dev_dataloader = DataLoader(data["dev"]["sents"], data["dev"]["labels"], data["dev"]["indexes"],
                            batch_size=args.batch_size, shuffle=False)

embedding_pre = []
if args.pretrained:
    print("use pretrained embedding")
    word2vec = {}
    for line in open("./embed_build/vec_300.txt"):
        word, vec = line.split()[0], line.split()[1:]
        word2vec[word] = map(eval, vec)
    unknown_pre = [1] * args.embedding_dim
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknown_pre)

model = Bilstm(args, embedding_pre).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    total_loss = 0
    for sents, labels, indexes in tqdm(train_dataloader, mininterval=1):
        sents, labels, indexes = sents.to(device), labels.to(device), indexes.to(device)
        out = model(sents, indexes, train_dataloader.length, device)
        loss = criterion(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss.item() / len(train_dataloader)


def valid():
    model.eval()
    count_predict = [0] * args.label_size
    count_total = [0] * args.label_size
    count_right = [0] * args.label_size
    for sents, labels, indexes in tqdm(dev_dataloader, mininterval=1):
        sents, labels, indexes = sents.to(device), labels.to(device), indexes.to(device)
        out = model(sents, indexes, dev_dataloader.length, device)
        out = torch.argmax(out, dim=1)
        for y1, y2 in zip(out, labels):
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_right[y1] += 1
    try:
        precision = sum(count_right[1:]) / sum(count_predict[1:])
    except:
        precision = 0

    try:
        recall = sum(count_right[1:]) / sum(count_total[1:])
    except:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0
    return precision, recall, f1


best_f1 = None
for epoch in range(args.num_epochs):
    loss = train()
    precision, recall, f1 = valid()
    print("-" * 90)
    print(
        "epoch[{} / {}], loss: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1: {:.5f}".format(epoch, args.num_epochs,
                                                                                             loss, precision, recall,
                                                                                             f1))
    print("-" * 90)

    if not best_f1 or best_f1 < f1:
        best_f1 = f1
        torch.save(model.state_dict(), args.save_best_model)
        print("best model has been saved.")

torch.save(model.state_dict(), args.save_final_model)
print("final model has been saved.")
