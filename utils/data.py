import copy
import json
from dataclasses import dataclass
from functools import reduce

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Vocabulary:
    """serve as candidates vocabulary"""
    def __init__(self, max_vocab: int = -1):
        self.UNK = "<UNK>"
        self.PADDING = "<PADDING>"
        self.stoi = {self.UNK: 0, self.PADDING: 1}
        self.itos = [self.UNK, self.PADDING]
        self.max_vocab = max_vocab

    def __len__(self):
        return len(self.itos)

    def save(self, path="./voc.txt"):
        with open(path, "w", encoding="utf-8") as f:
            for x in self.itos:
                f.write("%s\n" % x)
        print("Save vocabulary in", path)

    def load(self, path="./voc.txt"):
        self.stoi = {}
        self.itos = []
        with open(path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                self.itos.append(line)
                self.stoi[line] = count
                count += 1
        self.UNK, self.PADDING = self.itos[0:2]
        print("Loading vocabulary from", path)

    def update(self, token: str):
        if token not in self.itos:
            self.itos.append(token)
        else:
            pass

    def update_token(self, token: str):
        if token not in self.counts:
            self.counts[token] = 1
        else:
            self.counts[token] += 1

    def update_sentence(self, sentence: map):
        for token in sentence:
            self.update_token(token)

    def finalize_vocab(self, min_keep: int = 1, max_keep: int = 99999999):
        for k, v in sorted(self.counts.items(), key=lambda x: -x[1]):
            if min_keep <= v <= max_keep:
                if k not in self.stoi:
                    index = len(self.itos)
                    self.stoi[k] = index
                    self.itos.append(k)
        del self.counts

    def make_vocabulary(self, dataset: dict, key: str = 'candidates'):
        # temporary
        self.counts = {}
        dialog_paths = reduce(lambda x, y : x+y, dataset.values())
        for dialog_path in dialog_paths:
            with dialog_path.open("r") as fin:
                dialog_annotations = json.load(fin)
                features = []
                for dialog_annotation in dialog_annotations:
                    for turn_idx, turn in enumerate(dialog_annotation["turns"]):
                        candidates = reduce(lambda x, y: x + y, turn[key].values())
                        candidates = map(lambda x: x[-1], candidates)
                        self.update_sentence(candidates)

        self.finalize_vocab()


@dataclass
class DataIterator:
    dialog_paths: list
    vocab: Vocabulary

    def __post_init__(self):
        self.features, self.candidates = self.read_file()
        assert len(self.features) == len(self.candidates)

    def read_file(self):
        """Candidates and corresponding features are together."""
        all_features = []
        all_candidates = []
        for dialog_path in self.dialog_paths:
            feature_path = dialog_path.with_suffix(".pt")
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)["features"]
            all_features.extend(features)

            annotation_path = dialog_path.with_suffix(".annotation")
            with annotation_path.open("r") as fin:
                annotations = json.load(fin)
                for dialog_annotation in annotations:
                    for turn_annotation in dialog_annotation["turns"]:
                        candidates = reduce(lambda x, y: x + y, turn_annotation["candidates"].values())
                        candidates = list(map(lambda x: x[-1], candidates))
                        all_candidates.append(candidates)
        return all_features, all_candidates

    def __len__(self):
        return len(self.candidates)

    def batch_delivery(self, batch_size):
        dataset = TorchDataset(self.features, self.candidates)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch

    def add_padding(self, features, candidates):
        """Generate padded one-hot and contextualised embedding features and candidate masks."""

        len_list = [len(list(candi)) for candi in candidates]
        max_len = max(len_list)
        sorted_index = np.argsort(len_list)[::-1]
        fea_dim = features[sorted_index[0]][0].size()[0]

        oh = []
        ce = []
        masks = []
        candi_idxes = []
        for i, (candi, fea) in enumerate(zip(candidates, features)):
            pad_len = max_len - len_list[i]
            candi_idx = list(map(lambda x: self.vocab.stoi[x], candi))
            candi_idx.extend([self.vocab.stoi[self.vocab.PADDING] for _ in range(pad_len)])
            candi_idxes.append(candi_idx)
            oh.append(self.to_onehot(candi_idx))

            tmp_fea = copy.deepcopy(fea)
            tmp_fea.extend([torch.zeros(fea_dim) for _ in range(pad_len)])
            ce.append(torch.stack(tmp_fea, dim=0))
            masks.append([1. for _ in range(len_list[i])] + [0. for _ in range(max_len-len_list[i])])

        oh = torch.FloatTensor(oh).detach()
        ce = torch.stack(ce, dim=0).detach()
        masks = torch.FloatTensor(masks).detach()
        len_list = torch.LongTensor(len_list).detach()
        candi_idxes = torch.LongTensor(candi_idxes).detach()

        return oh, ce, masks, len_list, candi_idxes

    def to_onehot(self, data):
        """Generate one-hot features."""
        data = np.array(data)
        data = np.bincount(data, minlength=len(self.vocab))
        data[self.vocab.stoi[self.vocab.PADDING]] = 0.0
        data[self.vocab.stoi[self.vocab.UNK]] = 0.0
        return data.tolist()


@dataclass
class TorchDataset(Dataset):
    features: list
    candidates: list

    def __getitem__(self, index):
        return self.features[index], self.candidates[index]

    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.features) == len(self.candidates)
        return len(self.features)
