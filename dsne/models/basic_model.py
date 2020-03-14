import os
import pprint
import logging
import random
import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.metric import Loss, Accuracy
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.data import DataLoader

from ..networks import get_network
from ..datasets import get_dataset


class DomainAdaptationModel(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.cur_epoch, self.cur_iter = 0, 0
        self.is_train = True

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
        if self.cfg.DATA.GPU >= 0:
            self.ctx = [mx.gpu(self.cfg.DATA.GPU)]
        else:
            self.ctx = [mx.cpu()]

    def create_dataloader(self):
        train_dataset_params = self.cfg.DATA.TRAIN_DATASET_PARAMS
        train_dataset_params.update({'TARGET_PATH': self.cfg.META.SOURCE_PATH, 'TARGET_NUM': self.cfg.META.SOURCE_NUM})
        train_dataset = get_dataset(self.cfg.DATA.TRAIN_DATASET, train_dataset_params, self.cfg.DATA.TRAIN_TRANSFORM)

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True,
                                       last_batch='discard', num_workers=8, pin_memory=True)

        test_dataset_params = self.cfg.DATA.TEST_DATASET_PARAMS
        test_dataset_params.update({'TARGET_PATH': self.cfg.META.TARGET_PATH, 'TARGET_NUM': 0})
        test_dataset = get_dataset(self.cfg.DATA.TEST_DATASET, test_dataset_params, self.cfg.DATA.TEST_TRANSFORM, is_train=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=False,
                                      last_batch='keep', num_workers=8, pin_memory=True)

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
        self.meter = {'Xent-Src': Loss(name='XEnt-Src'), 'Acc-Src': Accuracy(name='Acc-Src')}
        if self.cfg.TRAIN.TRAIN_TARGET:
            self.meter.update({'Xent-Tgt': Loss(name='XEnt-Tgt'),'Acc-Tgt': Accuracy(name='Acc-Tgt')})

        self.eval_tracker = {'Epoch': 0, 'Iter': 0, 'Acc': 0}

    def reset_meter(self):
        for k, v in self.meter.items():
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

        self.summary()

    def train_epoch(self):
        self.is_train = True
        self.reset_meter()
        for data in self.train_loader:
            data = [d.as_in_context(self.ctx[0]) for d in data]

            with autograd.record():
                y_hat, embed = self.net(data[0])
                loss = self.criterion_xent(y_hat, data[1])

                self.meter['Xent-Src'].update(None, loss)
                self.meter['Acc-Src'].update(preds=[y_hat], labels=[data[1]])

                self.cur_iter += 1

                loss.backward()

            self.trainer.step(len(data[0]))

    def eval_epoch(self):
        self.is_train = False
        meter = Accuracy()
        meter.reset()

        for X, y in self.test_loader:
            X = X.as_in_context(self.ctx[0])
            y = y.as_in_context(self.ctx[0])

            y_hat, features = self.net(X)
            meter.update([y], [y_hat])

        acc = meter.get()[1]
        logging.info('Test  - Epoch {}, Iter {}, Acc {:.2f} %'.format(self.cur_epoch, self.cur_iter, acc * 100))

        if acc > self.eval_tracker['Acc']:
            self.eval_tracker.update({'Epoch': self.cur_epoch, 'Iter': self.cur_iter, 'Acc': acc})

        self.net.save_parameters('{}_{}_{}_{:.2f}.params'.format(self.cfg.META.CKPT_PATH, self.cur_epoch,
                                                                 self.cur_iter, acc))

    def log(self):
        msg = 'Train - Epoch {}, Iter {}'.format(self.cur_epoch, self.cur_iter)

        for k, mtc in self.meter.items():
            k, v = mtc.get()
            msg += ', {} {:.6f}'.format(k, v)

        logging.info(msg)

    def summary(self):
        logging.info('Best  - Epoch {}, Iter {}, Acc {:.2f}'.format(
            self.eval_tracker['Epoch'], self.eval_tracker['Iter'], self.eval_tracker['Acc'] * 100))
