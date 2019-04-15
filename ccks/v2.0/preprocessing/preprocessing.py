import random
from tqdm import tqdm
import numpy as np

prob = 0.5
batch_size = 128
test_size = 77092
res = batch_size - test_size % batch_size


def process_text(path):
    X, e1_list, e2_list = [], [], []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            _, e1, e2, words = line.strip().split("\t")
            X.append(words)
            e1_list.append(e1)
            e2_list.append(e2)
    return X, e1_list, e2_list


def process_label(path):
    y = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            label = line.strip().split("\t")[1].split()[0]
            y.append(label)
    return y


print("process train file...")
X_train, e1_train, e2_train = process_text("../data/sent_train.txt")
y_train = process_label("../data/sent_relation_train.txt")

# calculating truncated length
X_train_length = [len(x.split()) for x in X_train]
print("95% length - [{}]".format(np.percentile(X_train_length, 95)))

print("process dev file...")
X_dev, e1_dev, e2_dev = process_text("../data/sent_dev.txt")
y_dev = process_label("../data/sent_relation_dev.txt")

print("process test file...")
X_test, e1_test, e2_test = process_text("../data/sent_test.txt")

# #########################################
# reproduce train file, dev file. test file
# train
print("generating train.txt...")
train = zip(e1_train, e2_train, y_train, X_train)
with open("./train.txt", "w") as f:
    for e1, e2, y, x in train:
        if y != "0":
            f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")
        if y == "0" and random.random() > prob:
            f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")

# dev
print("generating dev.txt...")
dev = zip(e1_dev, e2_dev, y_dev, X_dev)
with open("./dev.txt", "w") as f:
    for e1, e2, y, x in dev:
        f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")

# test
print("generating test.txt...")
test = zip(e1_test + e1_test[:res], e2_test + e2_test[:res], X_test + X_test[:res])
with open("./test.txt", "w") as f:
    for e1, e2, x in test:
        f.write(e1 + "\t" + e2 + "\t" + x + "\n")

# ###################################################

# ###################################################
# generating word2vec txt
with open("../emb_build/train_vec.txt", "w") as f:
    for x in X_train + X_dev + X_test:
        f.write(x + "\n")
# ###################################################
