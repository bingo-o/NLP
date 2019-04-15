import torch
from torch import optim, nn
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description="LSTM text classification")
parser.add_argument(
    "--lr", type=float, default=1e-3, help="initial learning rate [default: 1e-3]")
parser.add_argument(
    "--num-epochs", type=int, default=100, help="number of epochs for train")
parser.add_argument("--batch-size", type=int, default=16,
                    help="batch size for training [default: 16]")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--cuda-able", action="store_true", default=False, help="enables cuda")
parser.add_argument("--save-model", type=str,
                    default="./LSTM_Text.pt", help="path to save the final model")
parser.add_argument(
    "--data", type=str, default="./data/corpus.pt", help="location of the data corpus")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="the probability for dropout (0 = no dropout) [default: 0.5]")
parser.add_argument("--embed-size", type=int, default=64,
                    help="number of embedding size [default: 64]")
parser.add_argument("--hidden-size", type=int, default=128,
                    help="number of lstm hidden size [default: 128]")
parser.add_argument(
    "--num-layers", type=int, default=3, help="biLSTM layer numbers")
parser.add_argument("--bidirectional", action="store_true",
                    help="if True, becomes a bidirectional LSTM [default: False]")

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
num_devices = torch.cuda.device_count()
device = torch.device("cuda:" + str(num_devices - 1) if use_cuda else "cpu")

# Load data
from data_loader import DataLoader
data = torch.load(args.data)
args.vocab_size = data["dict"]["vocab_size"]
args.label_size = data["dict"]["label_size"]

train_loader = DataLoader(
    data["train"]["sents"], data["train"]["labels"], batch_size=args.batch_size)
valid_loader = DataLoader(data["valid"]["sents"], data["valid"][
                          "labels"], batch_size=args.batch_size, shuffle=False)

# Build model
from model import LSTM_Text
model = LSTM_Text(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
# Train


def train():
    model.train()
    total_loss = 0
    for data, labels in tqdm(train_loader, mininterval=1):
        data = data.to(device)
        labels = labels.to(device)
        out = model(data)
        loss = criterion(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss.item() / train_loader.sents_size


def valid():
    model.eval()
    total_correct = 0
    for data, labels in tqdm(valid_loader, mininterval=1):
        data = data.to(device)
        labels = labels.to(device)
        out = model(data)
        total_correct += (torch.argmax(out, dim=1) == labels).sum()
    return total_correct.item() / valid_loader.sents_size

best_acc = None
for epoch in range(args.num_epochs):
    loss = train()
    acc = valid()
    print('-' * 90)
    print("epoch[{} / {}], loss: {:.9f}, acc: {:.4f}".format(epoch + 1, args.num_epochs, loss, acc))
    print('-' * 90)

    if not best_acc or best_acc < acc:
        best_acc = acc
        model_params = model.state_dict()

print("best_acc: " + str(best_acc))
torch.save(model_params, args.save_model)
