import argparse
import torch
from torch import optim, nn
from data_loader import DataLoader
from model import BilstmAtt
from const import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description="people relation extraction")
parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate [default: 5e-4]")
parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs for train")
parser.add_argument("--batch-size", type=int, default=128, help = "batch size for training [default: 128]")
parser.add_argument("--seed", type=int, default=1, help = "random seed")
parser.add_argument("--save-best-model", type=str, default="./best_model.pt", help = "path to save the best model")
parser.add_argument("--save-final-model", type=str, default="./final_model.pt",  help = "path to save the final model")
parser.add_argument("--data", type=str, default="./preprocessing/corpus.pt", help = "location of the data corpus")
parser.add_argument("--embedding-dim", type=int, default=300, help = "embedding dim [default: 300]")
parser.add_argument("--pos-dim", type=int, default=25, help = "pos dim [default: 25]")
parser.add_argument("--hidden-dim", type=int, default=200, help = "lstm hidden dim [default: 200]")
parser.add_argument("--num-layers", type=int, default=3, help = "biLSTM layer numbers")
parser.add_argument("--cuda-able", action="store_true", help = "enables cuda")
parser.add_argument("--pretrained", action="store_true", help = "use pretrained embedding")
parser.add_argument("--ensemble", action="store_true", help="whether to merge train file and dev file")

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
num_devices = torch.cuda.device_count()
device = torch.device("cuda:" + str(num_devices - 1) if use_cuda else "cpu")

data = torch.load(args.data)
args.vocab_size = data["dict"]["vocab_size"]
args.label_size = data["dict"]["label_size"]
word2id = data["dict"]["vocab"]
args.pos_size = 2 * LIMIT + 2

train_dataloader = DataLoader(data["train"]["sents"], data["train"]["pos1"], data["train"]["pos2"],
                              data["train"]["labels"], args.batch_size)
ensemble_dataloader = DataLoader(data["ensemble"]["sents"], data["ensemble"]["pos1"], data["ensemble"]["pos2"],
                                 data["ensemble"]["labels"], args.batch_size)
dev_dataloader = DataLoader(data["dev"]["sents"], data["dev"]["pos1"], data["dev"]["pos2"], data["dev"]["labels"],
                            args.batch_size, shuffle=False)

embedding_pre = []
if args.pretrained:
    print("use pretrained embedding")

    word2vec = {}
    for line in open("./emb_build/vec_300.txt"):
        word, vec = line.split()[0], line.split()[1:]
        word2vec[word] = map(eval, vec)

    unknown_pre = [1] * args.embedding_dim
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknown_pre)

model = BilstmAtt(args, embedding_pre).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


def train(dataloader):
    model.train()
    total_loss = 0

    for sents, pos1, pos2, labels in tqdm(dataloader, mininterval=1):
        sents, pos1, pos2, labels = sents.to(device), pos1.to(device), pos2.to(device), labels.to(device)
        out = model(sents, pos1, pos2, device)
        loss = criterion(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss.item() / len(dataloader)


def valid():
    model.eval()
    total = 0
    correct = 0
    count_predict = [0] * args.label_size
    count_total = [0] * args.label_size
    count_correct = [0] * args.label_size

    for sents, pos1, pos2, labels in tqdm(dev_dataloader, mininterval=1):
        sents, pos1, pos2, labels = sents.to(device), pos1.to(device), pos2.to(device), labels.to(device)
        out = model(sents, pos1, pos2, device)
        out = torch.argmax(out, dim=1)
        for y1, y2 in zip(out, labels):
            total += 1
            count_predict[y1] += 1
            count_total[y2] += 1
            if y1 == y2:
                count_correct[y1] += 1
                correct += 1

    try:
        precision = sum(count_correct[1:]) / sum(count_predict[1:])
    except:
        precision = 0

    try:
        recall = sum(count_correct[1:]) / sum(count_total[1:])
    except:
        recall = 0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0

    return correct / total, precision, recall, f1


if args.ensemble:
    dataloader = ensemble_dataloader
else:
    dataloader = train_dataloader

best_f1 = None
for epoch in range(args.num_epochs):
    loss = train(dataloader)
    acc, precision, recall, f1 = valid()

    print("-" * 90)
    print(
        "epoch[{} / {}], loss: {:.4f}, acc: {:.4f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}".format(epoch + 1, args.num_epochs,
                                                                                             loss, acc, precision,
                                                                                             recall, f1))
    print("-" * 90)

    if not best_f1 or best_f1 < f1:
        best_f1 = f1
        torch.save(model.state_dict(), args.save_best_model)
        print("best model has been saved.")

torch.save(model.state_dict(), args.save_final_model)
print("final model has been saved.")
