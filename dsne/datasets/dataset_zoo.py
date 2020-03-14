from mxnet.gluon.data.vision.transforms import Compose, Resize, Normalize, ToTensor
from .digit_dataset import DigitDataset


_DATASET = {'DIGIT': DigitDataset}


def get_transform(transform_params):
    transform = []
    if 'RESIZE' in transform_params:
        transform.append(Resize(transform_params['RESIZE']))

    transform.append(ToTensor())
    if 'MEAN' in transform_params and 'STD' in transform_params:
        transform.append(Normalize(transform_params['MEAN'], transform_params['STD']))

    transform = Compose(transform)
    return transform


def get_dataset(name, dataset_params, transform_params, is_train=True):
    transform = get_transform(transform_params)

    return _DATASET[name.upper()](dataset_params, is_train, transform)
