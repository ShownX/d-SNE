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


def load_bin(path):
    with open(path, 'rb') as fin:
        data = pk.load(fin)

    return data


class DigitDataset(Dataset):
    def __init__(self, cfg, transform=None, is_train=True):
        self.cfg = cfg
        self.transform = transform
        self.is_train = is_train

        self.init_dataset()

    def init_dataset(self):
        data = load_bin(self.cfg.TARGET_PATH)
        self.data = data['TR'] if self.is_train else data['TE']
        self.length = len(self.data[0])

    def __getitem__(self, idx):
        im, l = self.data[0][idx], self.data[1][idx]
        im = nd.array(im, dtype='float32')
        if self.transform is not None:
            im = self.transform(im)

        return im, l

    def __len__(self):
        return self.length


class DigitPairsDataset(Dataset):
    def __init__(self, cfg, transform=None, is_train=True):
        self.cfg = cfg

        self.ratio = 1
        self.transform = transform
        self.is_train = is_train
        self.init_dataset()

    def init_dataset(self):
        self.data_src = self.sample_data(self.cfg.SOURCE_PATH, self.cfg.SOURCE_NUM)
        self.data_tgt = self.sample_data(self.cfg.TARGET_PATH, self.cfg.TARGET_NUM)
        self.data_pairs = self.create_pairs()

    def create_pairs(self):
        pos_pairs, neg_pairs = [], []
        for ids, ys in enumerate(self.data_src[1]):
            for idt, yt in enumerate(self.data_tgt[1]):
                if ys == yt:
                    pos_pairs.append([ids, idt])
                else:
                    neg_pairs.append([ids, idt])

        if self.ratio > 0:
            random.shuffle(neg_pairs)
            pairs = pos_pairs + neg_pairs[: self.ratio * len(pos_pairs)]
        else:
            pairs = pos_pairs + neg_pairs

        random.shuffle(pairs)

        return pairs

    def sample_data(self, path, k):
        data = load_bin(path)['TR']

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

        return data

    def __getitem__(self, idx):
        """
        Override the function getitem
        :param idx: index
        :return:
        """
        idx1, idx2 = self.data_pairs[idx]
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

    def __len__(self):
        return len(self.data_pairs)

