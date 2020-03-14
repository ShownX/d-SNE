from mxnet import autograd
from mxnet.metric import Loss, Accuracy

from .basic_model import DomainAdaptationModel, SoftmaxCrossEntropyLoss
from ..networks import dSNELoss


class dSNEModel(DomainAdaptationModel):
    def create_meter(self):
        self.meter = {'Xent-Src': Loss(name='XEnt-Src'), 'Acc-Src': Accuracy(name='Acc-Src'),
                      'Xent-Tgt': Loss(name='XEnt-Tgt'), 'Acc-Tgt': Accuracy(name='Acc-Tgt'),
                      'Aux-Src': Loss(name='Aux-Src'), 'Aux-Tgt': Loss(name='Aux-Tgt'),
                      'Total-Src': Loss(name='Loss-Src'), 'Total-Tgt': Loss(name='Loss-Tgt')}
        self.eval_tracker = {'Epoch': 0, 'Iter': 0, 'Acc': 0}

    def create_criterion(self):
        self.criterion_xent = SoftmaxCrossEntropyLoss()

        dsne_params = self.get_lower_params(self.cfg.TRAIN.DSNE)
        dsne_params.update({'bs_src': self.cfg.DATA.BATCH_SIZE, 'bs_tgt': self.cfg.DATA.BATCH_SIZE,
                            'embed_size': self.cfg.NETWORK.NETWORK_PARAMS.FEATURE_SIZE})
        self.criterion_dsne = dSNELoss(**dsne_params)

    def train_epoch(self):
        self.is_train = True
        self.reset_meter()
        for data in self.train_loader:
            data = [d.as_in_context(self.ctx[0]) for d in data]

            if self.cfg.TRAIN.TRAIN_SOURCE:
                self.train_step(data[0], data[1], data[2], data[3], flag=False)

            if self.cfg.TRAIN.TRAIN_TARGET:
                self.train_step(data[2], data[3], data[0], data[1], flag=True)

            if self.cfg.TRAIN.LOG_ITV != 0 and self.cur_iter % self.cfg.TRAIN.LOG_ITV == 0:
                self.log()

        self.log()

    def train_step(self, Xs, Ys, Xt, Yt, flag=False):
        prefix = 'Tgt' if flag else 'Src'

        with autograd.record():
            ys_hat, fts = self.net(Xs)
            yt_hat, ftt = self.net(Xt)

            loss_xent = self.criterion_xent(ys_hat, Ys)
            loss_aux = self.criterion_dsne(fts, Ys, ftt, Yt)

            loss = (1 - self.cfg.TRAIN.ALPHA) * loss_xent + self.cfg.TRAIN.ALPHA * loss_aux

            self.meter['Xent-{}'.format(prefix)].update(None, loss_xent)
            self.meter['Acc-{}'.format(prefix)].update([Ys], [ys_hat])
            self.meter['Aux-{}'.format(prefix)].update(None, loss_aux)
            self.meter['Total-{}'.format(prefix)].update(None, loss)

            loss.backward()

            self.cur_iter += 1

        self.trainer.step(len(Xs))
