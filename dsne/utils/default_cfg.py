import os
import yaml
from easydict import EasyDict as edict


Cfg = edict()
# --------------------------------------------------------------- #
# Meta options
# --------------------------------------------------------------- #
Cfg.META = edict()
# Model
Cfg.META.MODEL = 'B'
# Classes corresponding labels
Cfg.META.CLASSES = None
# Source domain abbreviation
Cfg.META.SOURCE = ''
# Source domain data path
Cfg.META.SOURCE_PATH = ''
# Number of images used in the source domain, 0 use all images
Cfg.META.SOURCE_NUM_IMG = 0
# Target domain abbreviation
Cfg.META.TARGET = ''
# Target domain abbreviation
Cfg.META.TARGET_PATH = ''
# Number of images used in the target domain
Cfg.META.TARGET_NUM_IMG = 100
# Checkpoint directory
Cfg.META.CKPT_DIR = 'ckpts'
# Checkpoint prefix
Cfg.META.CKPT_PREFIX = None
# Checkpoint postfix
Cfg.META.CKPT_POSTFIX = None

# --------------------------------------------------------------- #
# Network options
# --------------------------------------------------------------- #
Cfg.NETWORK = edict()
# Network name
Cfg.NETWORK.NAME = 'LeNetPlus'
# Network initializer
Cfg.NETWORK.INIT = 'MSRAPrelu'
# Network resume weight path
Cfg.NETWORK.CKPT_PATH = ''
# Network Parameters
Cfg.NETWORK.PARAMS = edict()

# --------------------------------------------------------------- #
# Data options
# --------------------------------------------------------------- #
Cfg.DATA = edict()
# GPUS
Cfg.DATA.GPUS = [0]
# --------------------------------------------------------------- #
# Train options
# --------------------------------------------------------------- #
Cfg.TRAIN = edict()
# Training using source domain images
Cfg.TRAIN.TRAIN_SOURCE = True
# Training using target domain images
Cfg.TRAIN.TRAIN_TARGET = True
# Training random see
Cfg.TRAIN.RNG_SEED = 0
# Train maximum epoch
Cfg.TRAIN.MAX_EPOCH = 20
# Trainer
Cfg.TRAIN.OPTIM = 'SGD'
# Trainer optimizer
Cfg.TRAIN.OPTIM_PARAMS = edict()
# Learning rate
Cfg.TRAIN.OPTIM_PARAMS.LEARNING_RATE = 0.1
# weight decay
Cfg.TRAIN.OPTIM_PARAMS.WD = 1e-4
# Momentum
Cfg.TRAIN.OPTIM_PARAMS.MOMENTUM = 0.9


def update_config(args):
    if os.path.exists(args.cfg):
        with open(args.cfg, 'r') as fin:
            cfg = yaml.load(fin)

        if cfg is not None:
            for k in cfg.keys():
                Cfg[k].update(cfg[k])

        Cfg.META.MODEL = args.model
        Cfg.TRAIN.TRAIN_SOURCE = args.train_src
        Cfg.TRAIN.TRAIN_TARGET = args.train_tgt
        Cfg.TRAIN.CKPT_POSTFIX = args.session
        Cfg.NETWORK.CKPT_PATH = args.ckpt

        if Cfg.META.CKPT_PREFIX is None or len(Cfg.META.CKPT_PREFIX) == 0:
            Cfg.META.CKPT_PREFIX = '{}-{}'.format(Cfg.META.SOURCE, Cfg.META.TARGET)

        if Cfg.META.CKPT_POSTFIX is None or len(Cfg.META.CKPT_POSTFIX) == 0:
            Cfg.META.CKPT_POSTFIX = Cfg.META.CKPT_PREFIX
        else:
            Cfg.META.CKPT_POSTFIX = '{}-{}'.format(Cfg.META.CKPT_PREFIX, Cfg.META.CKPT_POSTFIX)

        Cfg.META.CKPT_PATH = os.path.join(Cfg.META.CKPT_DIR, Cfg.META.CKPT_PREFIX, Cfg.META.CKPT_POSTFIX)

        if not os.path.exists(os.path.dirname(Cfg.META.CKPT_PATH)):
            os.makedirs(os.path.dirname(Cfg.META.CKPT_PATH))

        return Cfg
    else:
        raise FileNotFoundError
