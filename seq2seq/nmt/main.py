import argparse
import torch
from torch import optim, nn
from model import Encoder, Decoder
from const import *
import random
import time
from utils import *
from tqdm import tqdm
from show import *

parser = argparse.ArgumentParser(description="Machine Translation")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--cuda-able", action="store_true", default=False, help="enables cuda")
parser.add_argument(
    "--data", type=str, default="./data/corpus.pt", help="location of the data corpus")
parser.add_argument("--hidden-size", type=int, default=256, help="hidden size")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--learning-rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--teacher-forcing-ratio", type=float,
                    default=0.5, help="teacher forcing ratio")
parser.add_argument("--num-iters", type=int, default=75000, help="num iters")
parser.add_argument(
    "--print-every", type=int, default=5000, help="every how many iters to print")

args = parser.parse_args()

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able
num_devices = torch.cuda.device_count()
device = torch.device("cuda:" + str(num_devices - 1) if use_cuda else "cpu")

data = torch.load(args.data)
max_len = data["max_len"]
input_lang_size = data["dict"]["input_lang_size"]
output_lang_id2word = data["dict"]["output_lang_id2word"]
input_lang_id2word = data["dict"]["input_lang_id2word"]
output_lang_size = data["dict"]["output_lang_size"]
pairs = [(torch.LongTensor(source), torch.LongTensor(target))
         for source, target in data["pairs"]]
training_pairs = [random.choice(pairs) for i in range(args.num_iters)]

encoder = Encoder(input_lang_size, args.hidden_size).to(device)
decoder = Decoder(
    output_lang_size, args.hidden_size, args.dropout, max_len).to(device)
criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)


def train(source, target):
    encoder_hidden = encoder.init_hidden().to(device)
    encoder_outputs = torch.zeros(max_len, args.hidden_size).to(device)
    loss = 0

    for i in range(len(source)):
        encoder_output, encoder_hidden = encoder(source[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.LongTensor([SOS]).to(device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random(
    ) < args.teacher_forcing_ratio else False

    if use_teacher_forcing:
        for i in range(len(target)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target[i])
            decoder_input = target[i]
    else:
        for i in range(len(target)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output, target[i])
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS:
                break
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / len(target)


def evaluate(source):
    with torch.no_grad():
        encoder_hidden = encoder.init_hidden().to(device)
        encoder_outputs = torch.zeros(max_len, args.hidden_size).to(device)
        for i in range(len(source)):
            encoder_output, encoder_hidden = encoder(source[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]

        decoder_input = torch.LongTensor([SOS]).to(device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_len, max_len)

        for i in range(max_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[i] = decoder_attention
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(output_lang_id2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:i + 1]

start = time.time()
total_loss = 0
losses = []

for iter in tqdm(range(1, args.num_iters + 1)):
    source, target = training_pairs[iter - 1]
    source = source.to(device).unsqueeze(1)
    target = target.to(device).unsqueeze(1)
    loss = train(source, target)
    total_loss += loss

    if iter % args.print_every == 0:
        avg_loss = total_loss / args.print_every
        losses.append(avg_loss)
        total_loss = 0
        print('%s (%d %d%%) %.4f' % (timeSince(
            start, iter / args.num_iters), iter, iter / args.num_iters * 100, avg_loss))
        
show_plot(losses)

# 测试
for i in range(6):
    source, target = random.choice(pairs)
    source = source.to(device)
    target = target.to(device)
    source_sent = " ".join(map(lambda x: input_lang_id2word[x], source.tolist()[:-1]))
    target_sent = " ".join(map(lambda x: output_lang_id2word[x], target.tolist()[:-1]))
    print('>', source_sent)
    print('=', target_sent)
    output_words, attentions = evaluate(source)
    output_sentence = " ".join(output_words)
    print('<', output_sentence)
    print()
    show_attention(source_sent, output_words, attentions)
    
torch.save(encoder, "./data/encoder.pt")
torch.save(decoder, "./data/decoder.pt")
