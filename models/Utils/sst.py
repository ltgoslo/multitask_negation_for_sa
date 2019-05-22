from collections import defaultdict
import os

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate

import torchtext
from torchtext.datasets import SST


class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)

        words = pack_sequence([w for w,_ in batch])
        targets = default_collate([t for _,t in batch])

        return words, targets


class ChallengeDataset(object):
    def __init__(self, vocab, lower_case, data_file="../data/challenge_dataset/sst-test.txt"):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(sequential=False)
        self.test = torchtext.data.TabularDataset(data_file, format="tsv", skip_header=False, fields=[("label", label), ("text", text)])
        self.test_split = [(torch.LongTensor(vocab.ws2ids(item.text)),
                       torch.LongTensor([int(item.label)])) for item in self.test]
    def get_split(self):
        return Split(self.test_split)



class SSTDataset(object):
    def __init__(self, vocab, lower_case, data_dir="../data/datasets/en/sst-fine"):

        self.vocab = vocab
        self.splits = {}

        for name in ["train", "dev", "test"]:
            filename = os.path.join(data_dir, name) + ".txt"
            self.splits[name] = self.open_split(filename, lower_case)

        x, y = zip(*self.splits["dev"])
        y = [int(i) for i in y]
        self.labels = sorted(set(y))

    def open_split(self, data_file, lower_case):
        text = torchtext.data.Field(lower=lower_case, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(sequential=False)
        data = torchtext.data.TabularDataset(data_file, format="tsv", skip_header=False, fields=[("label", label), ("text", text)])
        data_split = [(torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor([int(item.label)])) for item in data]
        return data_split


    def get_split(self, name):
        return Split(self.splits[name])
