"""
Script to pack the dataset
"""

import os
import numpy as np
import cv2
import struct
import pickle as pk


class Dataset(object):
    def __init__(self, root):
        self.root = root
        self.dataset = {}

    def read(self):
        raise NotImplementedError

    def dump(self, dump_path):
        with open(dump_path, 'wb') as fout:
            pk.dump(self.dataset, fout)


class MNIST(Dataset):
    def read(self):
        def read_img(path):
            with open(path, 'rb') as fin:
                magic, num, rows, cols = struct.unpack(">IIII", fin.read(16))
                img = np.fromfile(fin, dtype=np.uint8).reshape(-1, rows, cols)

            return img

        def read_lbl(path):
            with open(path, 'rb') as fin:
                magic, num = struct.unpack(">II", fin.read(8))
                lbl = np.fromfile(fin, dtype=np.int8)

            return lbl

        train_img = read_img(os.path.join(self.root, 'train-images-idx3-ubyte'))
        train_lbl = read_lbl(os.path.join(self.root, 'train-labels-idx1-ubyte'))

        test_img = read_img(os.path.join(self.root, 't10k-images-idx3-ubyte'))
        test_lbl = read_lbl(os.path.join(self.root, 't10k-labels-idx1-ubyte'))

        self.dataset = {'TR': [train_img, train_lbl], 'TE': [test_img, test_lbl]}


class MNISTM(Dataset):
    def read(self):
        def read_img_lbl(path, root):
            imgs, lbl = [], []
            with open(path, 'r') as fin:
                for line in fin.readlines():
                    path, lb = line.split()
                    im = cv2.cvtColor(cv2.imread(os.path.join(root, path)), cv2.COLOR_BGR2RGB)
                    imgs.append(im)
                    lbl.append(int(lb))
            
            imgs, lbl = np.array(imgs), np.array(lbl)
            
            return imgs, lbl

        train_img, train_lbl = read_img_lbl(os.path.join(self.root, 'mnist_m_train_labels.txt'),
                                            os.path.join(self.root, 'mnist_m_train'))

        test_img, test_lbl = read_img_lbl(os.path.join(self.root, 'mnist_m_test_labels.txt'),
                                          os.path.join(self.root, 'mnist_m_test'))

        self.dataset = {'TR': [train_img, train_lbl], 'TE': [test_img, test_lbl]}


_DATASET = {'MNIST': MNIST, 'MN': MNIST, 'MNIST-M': MNISTM, 'MM': MNISTM, 'MNISTM': MNISTM}


def get_packed_dataset(name):
    return _DATASET[name.upper()]
