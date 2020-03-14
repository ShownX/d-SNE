import random
from mxnet.gluon.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, batch_size, cls_idx_dict1, cls_idx_dict2, ratio=1):
        """
        Balanced Two Steam Sampler, use cls_idx_dict1 as main dictinary and list
        :param batch_size: batch size
        :param cls_idx_dict1: class index dictionary
        :param cls_idx_dict2: class index dictionary
        :param ratio: negative / positive flag
        """
        self.batch_size = batch_size
        self.cls_idx_dict1 = cls_idx_dict1
        self.cls_idx_dict2 = cls_idx_dict2
        self.ratio = ratio

        assert set(cls_idx_dict1.keys()) == set(cls_idx_dict2.keys()), 'The labels of two classes are not consistent'

        self.n_cls = len(cls_idx_dict1.keys())
        self.n_samples = self.batch_size // self.n_cls

        assert self.batch_size >= self.n_cls, "batch size should equal or larger than number of classes"

        self.length = self.cal_len()

    def balance_sampling(self):
        cls_idx_dict = {}
        for k, v in self.cls_idx_dict1.items():
            random.shuffle(v)
            cls_idx_dict[k] = {}
            cls_idx_dict[k]['v'] = v
            cls_idx_dict[k]['p'] = 0

        seq = []
        cls = []
        while len(seq) < self.length:
            for k, v in cls_idx_dict.items():
                m_pointer = min(v['p'] + self.n_samples, len(v['v']))
                seq.extend(v['v'][v['p']: m_pointer])
                cls.extend(list((k, )*(m_pointer-v['p'])))
                cls_idx_dict[k]['p'] = m_pointer

        return seq, cls

    def cal_len(self):
        length = 0
        for v in self.cls_idx_dict1.values():
            length += len(v)

        return length

    def ospg(self, idx_seq, cls_seq):
        # online sampling pair generation
        pairs = []
        for idx, cls in zip(idx_seq, cls_seq):

            rnd = random.uniform(0, 1)
            if rnd > 1. / (1 + self.ratio):
                # random select class
                cls_set = set(self.cls_idx_dict2.keys())
                cls_set.remove(cls)
                idy = random.randint(0, len(cls_set) - 1)
                yt = list(cls_set)[idy]
                # random select the negative samples
                idy = random.randint(0, len(self.cls_idx_dict2[yt]) - 1)
                idy = self.cls_idx_dict2[yt][idy]
            else:
                idy = random.randint(0, len(self.cls_idx_dict2[cls]) - 1)
                idy = self.cls_idx_dict2[cls][idy]

            pairs.append([idx, idy])

        return pairs

    def __iter__(self):
        idx_seq, cls_seq = self.balance_sampling()
        pairs = self.ospg(idx_seq, cls_seq)
        return iter(pairs)

    def __len__(self):
        return self.length


class RandomPairSampler(Sampler):
    def __init__(self, idx1, idx2):
        """
        Two Steam Random Sampler
        :param idx1: index 1
        :param idx2: index 2
        """
        self.lst1 = idx1
        self.lst2 = idx2

        self.length = len(idx1)

    def __iter__(self):
        random.shuffle(self.lst1)
        pairs = []
        for id1 in self.lst1:
            id2 = random.randint(0, len(self.lst2)-1)
            pairs.append([id1, id2])

        return iter(pairs)

    def __len__(self):
        return self.length
