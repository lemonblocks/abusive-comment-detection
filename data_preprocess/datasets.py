import torch
import torch.utils.data as tud
import pandas as pd
import torchtext
import numpy as np
from collections import Counter, OrderedDict


def build_vocab(train_csv=None, dev_csv=None, min_freq=2):
    '依据 train_csv 或 dev_csv 来构建词典'
    train_tokens = ' '.join(list(train_csv['tokenstr'])).split() if train_csv is not None else []
    dev_tokens = ' '.join(list(dev_csv['tokenstr'])).split() if dev_csv is not None else []

    # 构造 vocab 时把 train 与 dev 中的全部数据都拿来构造
    total_tokens = train_tokens + dev_tokens
    counter = Counter(total_tokens).most_common()
    ordered_dict = OrderedDict(counter)
    vocab = torchtext.vocab.vocab(ordered_dict, min_freq=2)

    special_tokens = ['<mask>', '<sep>', '<cls>', '<unk>', '<pad>']
    for token in special_tokens:
        vocab.insert_token(token, 0)
    
    vocab.set_default_index(vocab['<unk>'])

    return vocab


class ta_dataset(tud.Dataset):
    "tamil classification"

    def __init__(self, raw_csv, vocab, max_len):
        super().__init__()
        self.raw_csv = raw_csv
        self.tokens = raw_csv['tokenstr']
        self.tags = raw_csv['tag']
        self.vocab = vocab
        self.max_len = max_len

        self.label2tag = list(set(self.tags))
        self.tag2label = dict([(t, i) for i, t in enumerate(self.label2tag)])
        self.labels = [self.tag2label[t] for t in self.tags]
    
    def __len__(self):
        return self.raw_csv.shape[0]
    
    def __getitem__(self, index):
        raw_tokens = ['<cls>'] + self.tokens[index].split()
        raw_tokens = raw_tokens + ['<pad>'] * (self.max_len - len(raw_tokens))
        tokens = [self.vocab[t] for t in raw_tokens]
        label = self.labels[index]
        
        return torch.tensor(tokens).long(), torch.tensor(label).long()
