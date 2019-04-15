import argparse
import torch
from torch import nn, optim

parser = argparse.ArgumentParser(description="CNN Text Classification")
parser.add_argument(
    "--lr", type=float, default=1e-3, help="initial learning rate [default: 1e-3]")
parser.add_argument(
    "--num-epochs", type=int, default=100, help="number of epochs for train")
parser.add_argument(
    "--batch-size", type=int, default=16, help="batch size for training")
parser.add_argument("--save-model", type=str,
                    default="./CNN_Text.pt", help="path to save the final model")
parser.add_argument(
    "--data", type=str, default="./data/corpus.pt", help="location of the data corpus")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="the probability for dropout (0 = no dropout) [default: 0.5]")
parser.add_argument(
    "--embed-size", type=int, default=128, help="size of embedding [default: 128]")
parser.add_argument(
    "--num-kernels", type=int, default=128, help="number of each kind of kernel")
parser.add_argument(
    "--filter-sizes", type=str, default="3,4,5", help="filter sizes")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--cuda-able", action="store_true", default=False, help="enable cuda")

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
num_devices = torch.cuda.device_count()
device = torch.device("cuda:" + str(num_devices - 1) if use_cuda else "cpu")

# Load_data
from data_loader import DataLoader
data = torch.load(args.data)
args.vocab_size = data["dict"]["vocab_size"]
args.label_size = data["dict"]["label_size"]
args.filter_sizes = list(map(int, args.filter_sizes.split(",")))

train_loader = DataLoader(
    data["train"]["sents"], data["train"]["labels"], batch_size=args.batch_size)
valid_loader = DataLoader(
    data["valid"]["sents"], data["valid"]["labels"], batch_size=args.batch_size, shuffle=False)

# Build model
from model import CNN_Text
cnn_text = CNN_Text(args).to(device)
optimizer = optim.Adam(cnn_text.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
# Train


def train():
    cnn_text.train()
    total_loss = 0
    for data, labels in tqdm(train_loader, mininterval=1):
        data = data.to(device)
        labels = labels.to(device)
        out = cnn_text(data)
        loss = criterion(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss.item() / train_loader.sents_size

# Valid


def evaluate():
    cnn_text.eval()
    total_correct = 0
    for data, labels in tqdm(valid_loader, mininterval=1):
        data = data.to(device)
        labels = labels.to(device)
        out = cnn_text(data)
        total_correct += (torch.argmax(out, dim=1) == labels).sum()
    return total_correct.item() / valid_loader.sents_size

# Training and Save model
best_acc = None
for epoch in range(args.num_epochs):
    loss = train()
    acc = evaluate()
    print('-' * 90)
    print("epoch[{} / {}], loss: {:.9f}, acc: {:.4f}".format(epoch +
                                                             1, args.num_epochs, loss, acc))
    print('-' * 90)

    if not best_acc or best_acc < acc:
        best_acc = acc
        model_params = cnn_text.state_dict()
print("best_acc: " + str(best_acc))
torch.save(model_params, args.save_model)
