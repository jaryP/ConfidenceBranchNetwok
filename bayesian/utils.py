import numpy as np


def uniform(size, low=0, high=1):
    return np.random.uniform(low, high, size)


def normal(size, mean=0, std=1):
    return mean + std * np.random.randn(*size)


def get_init(name):
    name = name.lower()
    if name == 'uniform':
        return uniform
    elif name == 'normal':
        return normal
    else:
        assert False


