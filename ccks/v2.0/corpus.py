import os
import torch
from const import *
import argparse


class DictionarY:
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


class Words(DictionarY):
    def __init__(self):
        super(Words, self).__init__(WORD2ID, len(WORD2ID))

    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        for word in words:
            self.add_word(word)


class Labels(DictionarY):
    def __init__(self):
        super(Labels, self).__init__()

    def __call__(self, labels):
        labels = sorted(list(set(labels)), key=int)
        for label in labels:
            self.add_word(label)


class Corpus:
    def __init__(self, file_path, save_data, max_len, limit):
        self.train = os.path.join(file_path, "train.txt")
        self.dev = os.path.join(file_path, "dev.txt")
        self.ensemble = os.path.join(file_path, "ensemble.txt")
        self.save_data = save_data
        self.max_len = max_len
        self.limit = limit
        self.words = Words()
        self.labels = Labels()

    def process_sents(self, sents, word2id):
        sents = [[word2id[word] if word in word2id else UNK for word in sent] for sent in sents]
        sents = [sent + [PAD] * (self.max_len - len(sent)) for sent in sents]
        return sents

    def process_position(self, position, limit):
        def transform(num):
            if num < -limit:
                return 0
            if -limit <= num <= limit:
                return num + limit
            if num > limit:
                return 2 * limit

        position = [[transform(num) for num in pos] for pos in position]
        position = [pos + [2 * limit + 1] * (self.max_len - len(pos)) for pos in position]
        return position

    def parse_data(self, file, is_train=False, is_dev=False, is_ensemble=False):
        sents, pos1, pos2, labels = [], [], [], []
        for line in open(file):
            e1, e2, label, words = line.split("\t")
            e1_index, e2_index = words.index(e1), words.index(e2)
            words = words[:self.max_len]
            sents.append(words)
            index1, index2 = [], []
            for i in range(len(words)):
                index1.append(i - e1_index)
                index2.append(i - e2_index)
            pos1.append(index1)
            pos2.append(index2)
            labels.append(label)

        if is_train:
            print("process train file...")

            self.words(sents)
            self.labels(labels)

            self.train_sents = self.process_sents(sents, self.words.word2id)
            self.train_pos1 = self.process_position(pos1, self.limit)
            self.train_pos2 = self.process_position(pos2, self.limit)
            self.train_labels = [self.labels.word2id[label] for label in labels]

        if is_dev:
            print("process dev file...")

            self.dev_sents = self.process_sents(sents, self.words.word2id)
            self.dev_pos1 = self.process_position(pos1, self.limit)
            self.dev_pos2 = self.process_position(pos2, self.limit)
            self.dev_labels = [self.labels.word2id[label] for label in labels]

        if is_ensemble:
            print("process ensemble file...")

            self.ensemble_sents = self.process_sents(sents, self.words.word2id)
            self.ensemble_pos1 = self.process_position(pos1, self.limit)
            self.ensemble_pos2 = self.process_position(pos2, self.limit)
            self.ensemble_labels = [self.labels.word2id[label] for label in labels]

    def save(self):
        self.parse_data(self.train, is_train=True)
        self.parse_data(self.dev, is_dev=True)
        self.parse_data(self.ensemble, is_ensemble=True)

        data = {
            "max_len": self.max_len,
            "dict": {
                "vocab": self.words.word2id,
                "vocab_size": len(self.words),
                "label": self.labels.word2id,
                "label_size": len(self.labels)
            },
            "train": {
                "sents": self.train_sents,
                "pos1": self.train_pos1,
                "pos2": self.train_pos2,
                "labels": self.train_labels
            },
            "dev": {
                "sents": self.dev_sents,
                "pos1": self.dev_pos1,
                "pos2": self.dev_pos2,
                "labels": self.dev_labels
            },
            "ensemble": {
                "sents": self.ensemble_sents,
                "pos1": self.ensemble_pos1,
                "pos2": self.ensemble_pos2,
                "labels": self.ensemble_labels
            }
        }

        torch.save(data, self.save_data)
        print("Finishing dumping file to [{}]".format(self.save_data))
        print("vocab size - [{}]".format(len(self.words)))
        print("label size - [{}]".format(len(self.labels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="./preprocessing")
    parser.add_argument("--save-data", type=str, default="./preprocessing/corpus.pt")
    parser.add_argument("--max-len", type=int, default=34)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data, args.max_len, args.limit)
    corpus.save()

