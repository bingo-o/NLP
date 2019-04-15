import os
import torch
import argparse
from const import *

class Dictionary(object):

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
        labels = set(labels)
        for label in labels:
            self.add_word(label)


class Corpus(object):

    def __init__(self, file_path, save_data, max_len):
        self.train = os.path.join(file_path, "train")
        self.valid = os.path.join(file_path, "valid")
        self.save_data = save_data
        self.max_len = max_len
        self.words = Words()
        self.labels = Labels()

    def parse_data(self, file, is_train=True, fine_grained=False):
        sents, labels = [], []
        for line in open(file):
            label, _, words = line.replace("\xf0", " ").partition(" ")
            label = label.split(":")[0] if not fine_grained else label
            labels.append(label)

            words = words.strip().lower().split()
            if len(words) > self.max_len:
                words = words[:self.max_len]
            sents.append(words)
        if is_train:
            self.words(sents)
            self.labels(labels)
            self.train_sents = self.sen2id(sents, self.words.word2id)
            self.train_labels = [self.labels.word2id[label]
                                 for label in labels]
        else:
            self.valid_sents = self.sen2id(sents, self.words.word2id)
            self.valid_labels = [self.labels.word2id[label]
                                 for label in labels]

    def sen2id(self, sents, word2id):
    	return [[word2id[word] if word in word2id else UNK for word in sent] for sent in sents]


    def pad(self, sents):
        return [sent + [PAD] * (self.max_len - len(sent)) for sent in sents]

    def save(self):
        self.parse_data(self.train)
        self.parse_data(self.valid, False)

        self.train_sents = self.pad(self.train_sents)
        self.valid_sents = self.pad(self.valid_sents)

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
                "labels": self.train_labels
            },
            "valid": {
                "sents": self.valid_sents,
                "labels": self.valid_labels
            }
        }

        torch.save(data, self.save_data)

        print("Finish dumping data to file - [{}]".format(self.save_data))
        print("vocab size - [{}]".format(len(self.words)))
        print("label size - [{}]".format(len(self.labels)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Classification")
    parser.add_argument(
        "--file-path", type=str, default="./data/", help="file path")
    parser.add_argument(
        "--save-data", type=str, default="./data/corpus.pt", help="path to save processed data")
    parser.add_argument(
        "--max-len", type=int, default=16, help="max length of sentence")
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data, args.max_len)
    corpus.save()
