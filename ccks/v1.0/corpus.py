import os
import torch
import argparse
from const import *
from tqdm import tqdm


class Dictionary:
    def __init__(self, word2id={}, id=0):
        self.word2id = word2id
        self.id = id

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id += 1

    def convert(self):
        self.id2word = {id: word for word, id in self.word2id.items()}

    def __len__(self):
        return self.id


class Words(Dictionary):
    def __init__(self):
        super(Words, self).__init__(WORD2ID, len(WORD2ID))

    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        for word in words:
            self.add_word(word)


class Labels(Dictionary):
    def __init__(self):
        super(Labels, self).__init__()

    def __call__(self, labels):
        labels = sorted(list(set(labels)), key=int)
        for label in labels:
            self.add_word(label)


class Corpus:
    def __init__(self, file_path, save_data):
        self.train = os.path.join(file_path, "train.txt")
        self.dev = os.path.join(file_path, "dev.txt")
        self.save_data = save_data
        self.words = Words()
        self.labels = Labels()

    def process_sents(self, sents, word2id):
        sents = [[word2id[word] if word in word2id else UNK for word in sent] for sent in sents]
        return sents

    def parse_data(self, file, is_train=False, is_dev=False):
        sents, labels, indexes = [], [], []

        with open(file, "r") as f:
            for line in tqdm(f.readlines()):
                e1, e2, label, words = line.strip().split("\t")
                words = words.split()

                index = []
                for word in words:
                    if word == e1:
                        index.append(1)
                    elif word == e2:
                        index.append(2)
                    else:
                        index.append(0)
                sents.append(words)
                labels.append(label)
                indexes.append(index)

        if is_train:
            print("processing train file...")

            self.words(sents)
            self.labels(labels)

            train = zip(sents, labels, indexes)
            train = sorted(train, key=lambda x: len(x[0]), reverse=True)
            sents, labels, indexes = zip(*train)

            self.train_sents = self.process_sents(sents, self.words.word2id)
            self.train_labels = [self.labels.word2id[label] for label in labels]
            self.train_indexes = indexes

        if is_dev:
            print("processing dev file...")

            dev = zip(sents, labels, indexes)
            dev = sorted(dev, key=lambda x: len(x[0]), reverse=True)
            sents, labels, indexes = zip(*dev)

            self.dev_sents = self.process_sents(sents, self.words.word2id)
            self.dev_labels = [self.labels.word2id[label] for label in labels]
            self.dev_indexes = indexes

    def save(self):
        self.parse_data(self.train, is_train=True)
        self.parse_data(self.dev, is_dev=True)

        data = {
            "dict": {
                "vocab": self.words.word2id,
                "vocab_size": len(self.words),
                "label": self.labels.word2id,
                "label_size": len(self.labels)
            },
            "train": {
                "sents": self.train_sents,
                "labels": self.train_labels,
                "indexes": self.train_indexes
            },
            "dev": {
                "sents": self.dev_sents,
                "labels": self.dev_labels,
                "indexes": self.dev_indexes
            }
        }

        torch.save(data, self.save_data)

        print("Finishing dumping file to - [{}]".format(self.save_data))
        print("vocab size - [{}]".format(len(self.words)))
        print("label size - [{}]".format(len(self.labels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="./preprocessing")
    parser.add_argument("--save-data", type=str, default="./preprocessing/corpus.pt")
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data)
    corpus.save()