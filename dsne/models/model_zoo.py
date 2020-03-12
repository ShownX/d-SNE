from .basic_model import DomainAdaptationModel


_MODEL_ZOO = {'DOMAINADAPTATIONMODEL': DomainAdaptationModel,
              'BASICMODEL': DomainAdaptationModel,
              'B': DomainAdaptationModel,
              }


def get_model(cfg):
    return _MODEL_ZOO[cfg.META.MODEL](cfg=cfg)

