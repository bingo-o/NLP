import numpy as np
import torch


class DataLoader:
    def __init__(self, sents, pos1, pos2, labels, batch_size, shuffle=True):
        self.size = len(sents) - len(sents) % batch_size
        self.sents = np.asarray(sents[:self.size])
        self.pos1 = np.asarray(pos1[:self.size])
        self.pos2 = np.asarray(pos2[:self.size])
        self.labels = np.asarray(labels[:self.size])
        self.batch_size = batch_size

        self.step = 0
        self.stop_step = self.size // batch_size

        if shuffle:
            self.shuffle()

    def shuffle(self):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.sents = self.sents[indices]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return self.stop_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.batch_size
        bsz = self.batch_size
        self.step += 1

        sents = torch.LongTensor(self.sents[start:start + bsz])
        pos1 = torch.LongTensor(self.pos1[start:start + bsz])
        pos2 = torch.LongTensor(self.pos2[start:start + bsz])
        labels = torch.LongTensor(self.labels[start:start + bsz])

        return sents, pos1, pos2, labels
