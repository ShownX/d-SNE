import os
import pprint
import logging
import random
import numpy as np
import mxnet as mx
from mxnet.metric import Loss, Accuracy
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

from ..networks import get_network


class DomainAdaptationModel(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.cur_epoch, self.cur_iter = 0, 0

    def init_logging(self):
        filehandler = logging.FileHandler(self.cfg.META.CKPT_PATH + '.log')
        streamhandler = logging.StreamHandler()

        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)

        logger.info(pprint.pformat(self.cfg))

    def init_random_seed(self):
        if self.cfg.TRAIN.RNG_SEED > 0:
            random.seed(self.cfg.TRAIN.RNG_SEED)
            np.random.seed(self.cfg.TRAIN.RNG_SEED)
            mx.random.seed(self.cfg.TRAIN.RNG_SEED)

    def create_context(self):
        if isinstance(self.cfg.DATA.GPUS, list):
            self.ctx = [mx.gpu(gpu_id) for gpu_id in self.cfg.DATA.GPUS]
        elif self.cfg.DATA.GPUS > 0:
            self.ctx = [mx.gpu(self.cfg.DATA.GPUS)]
        else:
            self.ctx = [mx.cpu()]

    def create_dataloader(self):
        pass

    def create_criterion(self):
        self.criterion_xent = SoftmaxCrossEntropyLoss()

    def create_trainer(self):
        optim_params = self.get_lower_params(self.cfg.TRAIN.OPTIM_PARAMS)
        self.trainer = mx.gluon.Trainer(params=self.net.collect_params(),
                                        optimizer=self.cfg.TRAIN.OPTIM.lower(),
                                        optimizer_params=optim_params)

        if self.cfg.NETWORK.CKPT_PATH is not None and os.path.exists(self.cfg.NETWORK.CKPT_PATH.replace('.params', '.states')):
            logging.info('Loading optimizer from {}'.format(self.cfg.NETWORK.CKPT_PATH.replace('.params', '.states')))
            self.trainer.load_states(self.cfg.NETWORK.CKPT_PATH.replace('.params', '.states'))

    def create_meter(self):
        self.meters = {'Xent-Src': Loss(), 'Xent-Tgt': Loss(), 'Acc-Src': Accuracy(), 'Acc-Tgt': Accuracy(),
                       'Aux-Src': Loss(), 'Aux-Tgt': Loss(), 'Total-Src': Loss(), 'Total-Tgt': Loss()}

        self.eval_tracker = {'Epoch': 0, 'Iter': 0, 'Acc': 0}

    def reset_meter(self):
        for k, v in self.meters.items():
            v.reset()

    @staticmethod
    def get_lower_params(params_dict):
        params = {}
        for k, v in params_dict.items():
            params[k.lower()] = v

        return params

    def create_net(self):
        network_params = self.get_lower_params(self.cfg.NETWORK.PARAMS)
        self.net = get_network(self.cfg.NETWORK.NAME, network_params)

        if self.cfg.NETWORK.CKPT_PATH is None or len(self.cfg.NETWORK.CKPT_PATH) == 0:
            self.net.initialize(self.cfg.NETWORK.INIT)
        else:
            logging.info('Loading weights from {}'.format(self.cfg.NETWORK.CKPT_PATH))
            self.net.load_parameters(self.cfg.NETWORK.CKPT_PATH)

            epoch_iter_acc = os.path.splitext(os.path.basename(self.cfg.NETWORK.CKPT_PATH))[0].split('_')[-3:]
            self.cur_epoch, self.cur_iter = int(epoch_iter_acc[0]), int(epoch_iter_acc[1])
            acc = float(epoch_iter_acc[2])
            self.eval_tracker.update({'Epoch': self.cur_epoch, 'Iter': self.cur_iter, 'Acc': acc})

        self.net.collect_params().reset_ctx(self.ctx)

    def train(self):
        self.init_logging()
        self.init_random_seed()

        self.create_context()
        self.create_dataloader()

        self.create_net()
        self.create_criterion()
        self.create_trainer()
        self.create_meter()

        for epoch in range(self.cur_epoch + 1, self.cfg.TRAIN.MAX_EPOCH + 1):
            self.cur_epoch = epoch
            self.train_epoch()
            self.eval_epoch()

    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass

    def save_epoch(self):
        pass

    def test(self):
        pass
