from abc import ABC, abstractmethod

from torch import nn


class BranchModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def n_branches(self):
        raise NotImplemented