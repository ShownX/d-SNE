from .lenetplus import LeNetPlus


_Networks = {'LENETPLUS': LeNetPlus, 'L': LeNetPlus}


def get_network(name, network_params):
    return _Networks[name.upper()](**network_params)

