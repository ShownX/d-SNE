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
# Target domain abbreviation
Cfg.META.TARGET = ''
# Target domain abbreviation
Cfg.META.TARGET_PATH = ''
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
# Network initializer, e.g. 'MSRAPrelu'
Cfg.NETWORK.INIT = None
# Network resume weight path
Cfg.NETWORK.CKPT_PATH = ''
# Network Parameters
Cfg.NETWORK.PARAMS = edict()

# --------------------------------------------------------------- #
# Data options
# --------------------------------------------------------------- #
Cfg.DATA = edict()
# GPUS
Cfg.DATA.GPU = 0
# Data batch size
Cfg.DATA.BATCH_SIZE = 256
# TRAIN Dataset
Cfg.DATA.TRAIN_DATASET = 'DIGITS'
Cfg.DATA.TRAIN_DATASET_PARAMS = edict()
# Number of images used in the source domain, 0 use all images
Cfg.DATA.TRAIN_DATASET_PARAMS.TRAIN_SOURCE_NUM = 0
# Number of images used in the source domain, 0 use all images, 100 for 10 images / class
Cfg.DATA.TRAIN_DATASET_PARAMS.TRAIN_TARGET_NUM = 100
# Data sample
Cfg.DATA.TRAIN_DATASET_PARAMS.SAMPLE_RATIO = 3
# Train Transform
Cfg.DATA.TRAIN_TRANSFORM = edict()
# Image resize
Cfg.DATA.TRAIN_TRANSFORM.RESIZE = 32
# Image normalize
Cfg.DATA.TRAIN_TRANSFORM.MEAN = 0.5
# Image normalize
Cfg.DATA.TRAIN_TRANSFORM.STD = 0.5

# Test Dataset
Cfg.DATA.TEST_DATASET = 'DIGITS'
Cfg.DATA.TEST_TRANSFORM = edict()
# Image resize
Cfg.DATA.TEST_TRANSFORM.RESIZE = 32
# Image normalize
Cfg.DATA.TEST_TRANSFORM.MEAN = 0.5
# Image normalize
Cfg.DATA.TEST_TRANSFORM.STD = 0.5
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
Cfg.TRAIN.MAX_EPOCH = 10
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
# Log Iterval
Cfg.TRAIN.LOG_ITV = 50
# dSNE
Cfg.TRAIN.DSNE = edict()
# Margin
Cfg.TRAIN.DSNE.MARGIN = 1
# Feature normalization
Cfg.TRAIN.DSNE.FN = True
# Alpha
Cfg.TRAIN.ALPHA = 0.1


def update_config(args):
    if os.path.exists(args.cfg):
        with open(args.cfg, 'r') as fin:
            cfg = yaml.safe_load(fin)

        if cfg is not None:
            for k in cfg.keys():
                Cfg[k].update(cfg[k])

        Cfg.META.MODEL = args.model
        Cfg.TRAIN.TRAIN_SOURCE = args.train_src
        Cfg.TRAIN.TRAIN_TARGET = args.train_tgt
        Cfg.META.CKPT_POSTFIX = args.session
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
