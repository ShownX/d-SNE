from .lenetplus import LeNetPlus
from .conv2 import Conv2

_Networks = {'LENETPLUS': LeNetPlus, 'L': LeNetPlus,
             'CONV2': Conv2, 'C': Conv2}


def get_network(name, network_params):
    return _Networks[name.upper()](**network_params)

