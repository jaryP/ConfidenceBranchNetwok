from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions import Beta, Bernoulli
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from bayesian.utils import get_init


class BayesianPosterior(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_posterior(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplemented


class MatrixEmbedding(BayesianPosterior):
    def __init__(self, size: int,
                 distribution: str = 'beta',
                 initializer: str = 'uniform'):
        super().__init__()

        distribution = distribution.lower()
        if distribution == 'beta':
            size = (size, 2)
        elif distribution == 'bernoulli':
            size = (size, 1)
        else:
            assert False

        init = get_init(initializer)
        matrix = torch.tensor(init(size))

        # if distribution == 'beta' or distribution == 'uniform':
        #     matrix = torch.sigmoid(matrix)

        self.distribution = distribution
        self.matrix = nn.Parameter(matrix)

    def get_posterior(self, **kwargs):
        if self.distribution == 'beta':
            a, b = torch.chunk(torch.relu(self.matrix), 2, dim=1)
            a, b = a.clamp(0.1, 5), b.clamp(0.1, 5)
            return Beta(a.view(-1), b.view(-1))
        elif self.distribution == 'bernoulli':
            return Bernoulli(probs=torch.sigmoid(self.matrix))

    def __call__(self, branch_index, sample_shape=1, *args, **kwargs):
        if self.distribution == 'bernoulli':
            return torch.bernoulli(torch.sigmoid(self.matrix))[branch_index]
        sampled = self.get_posterior().rsample([sample_shape]).mean(0)
        return sampled[branch_index]


class LayerEmbeddingBeta(BayesianPosterior):
    def __init__(self, alpha_layers, beta_layers, min_clamp=1e-10, max_clamp=5):
        super().__init__()

        self.beta_layer = beta_layers
        self.alpha_layers = alpha_layers
        self.min_clamp = max(1e-10, min_clamp) if min_clamp \
                                                  is not None else 1e-10
        self.max_clamp = max(self.min_clamp, max_clamp) if max_clamp \
                                                           is not None else None

        # betas = []
        # for o in range(n_branches):
        #     # f = torch.flatten(o, 1)
        #     betas.append(nn.Sequential(nn.Flatten(),
        #                                nn.Linear(input_size, 1),
        #                                nn.Sigmoid()))
        #
        # distribution = distribution.lower()
        # if distribution == 'beta':
        #     size = (size, 2)
        # elif distribution == 'bernoulli':
        #     size = (size, 1)
        # else:
        #     assert False
        #
        # init = get_init(initializer)
        # matrix = torch.tensor(init(size))
        #
        # # if distribution == 'beta' or distribution == 'uniform':
        # #     matrix = torch.sigmoid(matrix)
        #
        # self.distribution = distribution
        # self.matrix = nn.Parameter(matrix)

    def get_posterior(self, logits, branch_index, **kwargs):
        a = self.alpha_layers[branch_index](logits)
        b = self.beta_layer[branch_index](logits)
        # a, b = ab.chunk(2, dim=1)
        a = a.clamp(self.min_clamp, self.max_clamp)
        b = b.clamp(self.min_clamp, self.max_clamp)
        # a, b = a.clamp(0.1), b.clamp(0.1)
        # a, b = a + 1, b + 0.1
        distribution = Beta(a, b)
        return distribution

    def __call__(self, logits, branch_index, sample_shape=1, *args, **kwargs):
        # ab = self.beta_layer[branch_index](logits)
        # a, b = ab.chunk(2, dim=1)
        # distribution = Beta(a, b)
        # return distribution.rsample([sample_shape]).mean(0)
        post = self.get_posterior(logits=logits, branch_index=branch_index)
        return post.rsample([sample_shape]).mean(0)


class LayerEmbeddingContBernoulli(BayesianPosterior):
    def __init__(self, alpha_layers):
        super().__init__()
        self.alpha_layers = alpha_layers

    def get_posterior(self, logits, branch_index, **kwargs):
        a = self.alpha_layers[branch_index](logits)
        # a, b = ab.chunk(2, dim=1)
        # a = a.clamp(self.min_clamp, self.max_clamp)
        # b = b.clamp(self.min_clamp, self.max_clamp)
        # a, b = a.clamp(0.1), b.clamp(0.1)
        # a, b = a + 1, b + 0.1
        distribution = ContinuousBernoulli(a)
        return distribution

    def __call__(self, logits, branch_index, sample_shape=1, *args, **kwargs):
        # ab = self.beta_layer[branch_index](logits)
        # a, b = ab.chunk(2, dim=1)
        # distribution = Beta(a, b)
        # return distribution.rsample([sample_shape]).mean(0)
        post = self.get_posterior(logits=logits, branch_index=branch_index)
        return post.rsample([sample_shape]).mean(0)
