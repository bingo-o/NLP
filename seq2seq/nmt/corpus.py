import unicodedata
import re
import os
import argparse
import torch
from const import *


class Dictionary(object):

    def __init__(self, name):
        self.name = name
        self.word2id = {"<sos>": SOS, "<eos>": EOS}
        self.id2word = {SOS: "<sos>", EOS: "<eos>"}
        self.word2count = {}
        self.id = len(self.word2id)

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id2word[self.id] = word
            self.word2count[word] = 1
            self.id += 1
        else:
            self.word2count[word] += 1
    
    def __len__(self):
        return self.id
    
    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        for word in words:
            self.add_word(word)


class Corpus(object):

    def __init__(self, file_path, lang1, lang2, save_data, max_len):
        self.file_path = file_path
        self.file = os.path.join(file_path, "%s-%s.txt" % (lang1, lang2))
        self.save_data = save_data
        self.max_len = max_len
        self.input_lang = Dictionary(lang1)
        self.output_lang = Dictionary(lang2)

    def unicode2Ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicode2Ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def parse_data(self, file, reverse=False):
        pairs = []
        print("Reading lines...")
        for line in open(file, "r", encoding="utf-8"):
            source, target = line.strip().split("\t")
            source = self.normalizeString(source)
            target = self.normalizeString(target)
            pairs.append([source, target])
            
        if reverse:
            pairs = [list(reversed(pair)) for pair in pairs]
            self.input_lang, self.output_lang = self.output_lang, self.input_lang
        
        print("Read %s sentence pairs" % len(pairs))
        
        eng_prefixes = (
            "i am", "i m",
            "he is", "he s",
            "she is", "she s",
            "you are", "you re",
            "we are", "we re",
            "they are", "they re"
        )
        
        pairs = list(filter(lambda p: len(p[0].split()) < self.max_len and len(
            p[1].split()) < self.max_len and p[1].startswith(eng_prefixes), pairs))
        
        print("Trimmed to %s sentence pairs" % len(pairs))
        
        with open(os.path.join(self.file_path, "pairs.txt"), "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair[0] + "\t" + pair[1] + "\n")
        
        sources, targets = zip(*pairs)
        sources = list(map(lambda x: x.split(), sources))
        targets = list(map(lambda x: x.split(), targets))
        
        self.input_lang(sources)
        self.output_lang(targets)
        sources = self.sen2id(sources, self.input_lang.word2id)
        targets = self.sen2id(targets, self.output_lang.word2id)
        self.pairs = list(zip(sources, targets))
    
    def sen2id(self, sents, word2id):
        return [[word2id[word] for word in sent] + [EOS] for sent in sents]
    
    def save(self):
        self.parse_data(self.file, True)
        
        data = {
            "max_len": self.max_len,
            "dict": {
                "input_lang_id2word": self.input_lang.id2word,
                "input_lang_size": len(self.input_lang),
                "output_lang_id2word": self.output_lang.id2word,
                "output_lang_size": len(self.output_lang)
            },
            "pairs": self.pairs
        }
        
        torch.save(data, self.save_data)
        
        print("Counted words:")
        print(self.input_lang.name, len(self.input_lang))
        print(self.output_lang.name, len(self.output_lang))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="./data/", help="file path")
    parser.add_argument("--lang1", type=str, default="eng", help="input lang")
    parser.add_argument("--lang2", type=str, default="fra", help="output lang")
    parser.add_argument("--save-data", type=str, default="./data/corpus.pt", help="path to save the processed data")
    parser.add_argument("--max-len", type=int, default=10, help="max length of sentence")
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.lang1, args.lang2, args.save_data, args.max_len)
    corpus.save()