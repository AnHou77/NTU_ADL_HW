from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch

from nltk.tokenize import word_tokenize

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        type: str
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.type = type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        split_data = []
        labels = []
        ids = []
        for data in samples:
            split_data.append(word_tokenize(data['text']))
            if self.type == 'train':
                labels.append(self.label2idx(data['intent']))
            ids.append(data['id'])
        vocabs = self.vocab.encode_batch(split_data, to_len=self.max_len)
        if self.type == 'train':
            return {'data': torch.tensor(vocabs), 'label': torch.tensor(labels), 'id': ids}
        return {'data': torch.tensor(vocabs), 'id': ids}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class TagClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        type: str
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.type = type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        split_data = []
        ids = []
        seq_len = []
        labels = torch.zeros((len(samples),self.max_len),dtype=torch.long)
        i = 0
        for data in samples:
            split_data.append(data['tokens'])
            if self.type == 'train':
                j = 0
                for tag in data['tags']:
                    labels[i][j] = self.label2idx(tag)
                    j += 1
                i += 1
            ids.append(data['id'])
            seq_len.append(len(data['tokens']))
        vocabs = self.vocab.encode_batch(split_data, to_len=self.max_len)
        if self.type == 'train':
            return {'data': torch.tensor(vocabs), 'label': labels, 'id': ids, 'seq_len': seq_len}
        return {'data': torch.tensor(vocabs), 'id': ids, 'seq_len': seq_len}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]