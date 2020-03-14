import os
import random
import numpy as np
import pickle as pk
from mxnet import nd
from mxnet.gluon.data.dataset import Dataset


def _gen_cls_idx_dict(idx_cls_lst):
    cls_idx_dict = {}
    for idx, y in enumerate(idx_cls_lst):
        y = int(y)
        if y in cls_idx_dict:
            cls_idx_dict[y].append(idx)
        else:
            cls_idx_dict[y] = [idx]
    return cls_idx_dict


class DigitDataset(Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        self.cfg = cfg

        self.is_train = is_train

        self.transform = transform
        self.init_dataset()

    @staticmethod
    def load_bin(path):
        with open(path, 'rb') as fin:
            data = pk.load(fin)

        return data

    def init_dataset(self):
        if self.is_train:
            self.data_src, self.idx_src = self.sample_data(self.cfg.SOURCE_PATH, self.cfg.SOURCE_NUM)
            self.data_tgt, self.idx_tgt = self.sample_data(self.cfg.TARGET_PATH, self.cfg.TARGET_NUM)
        else:
            self.data_tgt, self.idx_tgt = self.sample_data(self.cfg.TARGET_PATH, self.cfg.TARGET_NUM)

    def create_pairs(self):
        pos_pairs, neg_pairs = [], []
        for ids, ys in enumerate(self.arrs[1]):
            for idt, yt in enumerate(self.arrt[1]):
                if ys == yt:
                    pos_pairs.append([ids, ys, idt, yt, 1])
                else:
                    neg_pairs.append([ids, ys, idt, yt, 0])

        if self.ratio > 0:
            random.shuffle(neg_pairs)
            pairs = pos_pairs + neg_pairs[: self.ratio * len(pos_pairs)]
        else:
            pairs = pos_pairs + neg_pairs

        random.shuffle(pairs)

        return pairs

    def sample_data(self, path, k):
        data = self.load_bin(path)
        data = data['TR'] if self.is_train else data['TE']

        if k > 0:
            # each class has equivalent samples
            x, y = data
            classes = np.unique(y)
            classes_counts = {c: sum(y == c) for c in classes}
            classes_idx = {}
            for c in classes:
                classes_idx[c] = np.where(y == c)[0]

            num_class = len(classes)
            num_sample_per_class = k // num_class

            num_selected = 0
            classes_selected = {}
            # sampling
            for c in classes:
                random.shuffle(classes_idx[c])
                classes_selected[c] = classes_idx[c][: min(num_sample_per_class, classes_counts[c])]
                num_selected += classes_selected[c]

            idx_selected = np.array([idx for idx in classes_selected.values()]).ravel()

            x = x[idx_selected]
            y = y[idx_selected].ravel().astype('int32')

            data = [x, y]

        # cls_idx = _gen_cls_idx_dict(data[1])
        idx = list(range(len(data[1])))

        return data, idx

    def __getitem__(self, idx):
        """
        Override the function getitem
        :param idx: index
        :return:
        """
        if self.is_train:
            idx1, idx2 = idx
            im1, l1 = self.data_src[0][idx1], self.data_src[1][idx1]
            im2, l2 = self.data_tgt[0][idx2], self.data_tgt[1][idx2]

            im1 = nd.array(im1, dtype='float32')
            im2 = nd.array(im2, dtype='float32')

            if self.transform is not None:
                im1 = self.transform(im1)

            if self.transform is not None:
                im2 = self.transform(im2)

            lc = 1 if l1 == l2 else 0

            return im1, l1, im2, l2, lc
        else:
            im, l = self.data_tgt[0][idx], self.data_tgt[1][idx]
            im = nd.array(im, dtype='float32')
            if self.transform is not None:
                im = self.transform(im)

            return im, l

    def __len__(self):
        if self.is_train:
            return len(self.idx_src)
        else:
            return len(self.idx_tgt)
