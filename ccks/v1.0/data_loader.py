import numpy as np
import torch
from collections import Counter


class DataLoader:
    def __init__(self, sents, labels, indexes, batch_size, shuffle=True):
        self.sents = np.asarray(sents)
        self.sents_size = len(sents)
        self.labels = np.asarray(labels)
        self.indexes = np.asarray(indexes)
        self.batch_size = batch_size

        self.step = 0
        self.stop_step = (self.sents_size - 1) // batch_size + 1

        if shuffle:
            self.shuffle()

    def shuffle(self):
        indices = np.arange(self.sents_size)
        np.random.shuffle(indices)
        self.sents = self.sents[indices]
        self.labels = self.labels[indices]
        self.indexes = self.indexes[indices]

    def __len__(self):
        return self.stop_step

    def __iter__(self):
        return self

    def weight(self):
        count = sorted(Counter(self.labels).items(), key=lambda x: x[0])
        weight = []
        for item in count:
            weight.append(self.sents_size / item[1])
        weight = torch.softmax(torch.Tensor(weight), dim=0)
        return weight

    def pad(self, X):
        self.length = [len(x) for x in X]
        max_len = np.max(self.length)
        return [x + [0] * (max_len - len(x)) for x in X]

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.batch_size
        bsz = self.batch_size

        if self.batch_size > (self.sents_size - start):
            self.step = 0
            raise StopIteration()

        self.step += 1

        sents = torch.LongTensor(self.pad(self.sents[start:start + bsz]))
        labels = torch.LongTensor(self.labels[start:start + bsz])
        indexes = torch.LongTensor(self.pad(self.indexes[start:start + bsz]))
        return sents, labels, indexes
