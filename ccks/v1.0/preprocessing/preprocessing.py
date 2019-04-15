from tqdm import tqdm
import random


def process_text(path):
    X, e1_list, e2_list = [], [], []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            _, e1, e2, words = line.strip().split("\t")
            e1_list.append(e1)
            e2_list.append(e2)
            X.append(words)
    return X, e1_list, e2_list


def process_label(path):
    y = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            y.append(line.strip().split("\t")[1].split()[0])
    return y


print("processing train file...")
X_train, e1_train, e2_train = process_text("../data/sent_train.txt")
y_train = process_label("../data/sent_relation_train.txt")

print("processing dev file...")
X_dev, e1_dev, e2_dev = process_text("../data/sent_dev.txt")
y_dev = process_label("../data/sent_relation_dev.txt")

print("processing test file...")
X_test, e1_test, e2_test = process_text("../data/sent_test.txt")

# ######################################################
print("generating word vector txt...")

X = X_train + X_dev + X_test
with open("../embed_build/train_vec.txt", "w") as f:
    f.writelines([x + "\n" for x in X])
# #######################################################

# Reproducing train file, dev file, test file
# train
print("generating train.txt...")
with open("train.txt", "w") as f:
    count_zero = 0
    for e1, e2, y, x in zip(e1_train, e2_train, y_train, X_train):
        if y != "0":
            f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")
        if y == "0" and random.random() > 0.5 and count_zero < 38501:
            f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")
            count_zero += 1

# dev
print("generating dev.txt...")
with open("dev.txt", "w") as f:
    for e1, e2, y, x in zip(e1_dev, e2_dev, y_dev, X_dev):
        f.write(e1 + "\t" + e2 + "\t" + y + "\t" + x + "\n")

# test
print("generating test.txt...")
with open("test.txt", "w") as f:
    for e1, e2, x in zip(e1_test, e2_test, X_test):
        f.write(e1 + "\t" + e2 + "\t" + x + "\n")
