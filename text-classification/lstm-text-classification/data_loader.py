import numpy as np
import torch


class DataLoader(object):

    def __init__(self, sents, labels, batch_size, shuffle=True):
        self.sents = np.asarray(sents)
        self.sents_size = len(sents)
        self.labels = np.asarray(labels)
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

    def __len__(self):
        return self.stop_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()
        start = self.step * self.batch_size
        bsz = min(self.batch_size, self.sents_size - start)
        self.step += 1
        data = torch.LongTensor(self.sents[start:start+bsz])
        labels = torch.LongTensor(self.labels[start:start+bsz])
        return data, labels
