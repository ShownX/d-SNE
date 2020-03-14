from .basic_model import DomainAdaptationModel
from .dsne_model import dSNEModel


_MODEL_ZOO = {'DOMAINADAPTATIONMODEL': DomainAdaptationModel,
              'BASICMODEL': DomainAdaptationModel,
              'B': DomainAdaptationModel,
              'DSNE': dSNEModel,
              'D-SNE': dSNEModel,
              'D': dSNEModel
              }


def get_model(cfg):
    return _MODEL_ZOO[cfg.META.MODEL](cfg=cfg)

