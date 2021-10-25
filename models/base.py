from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from operator import mul

import torch
from torch import nn


class IntermediateBranch(nn.Module):
    def __init__(self, classifier: nn.Module,
                 preprocessing: nn.Module = None):
        super().__init__()
        self.preprocessing = preprocessing if preprocessing is not None \
            else lambda x: x

        self.classifier = classifier

    def preprocess(self, x):
        return self.preprocessing(x)

    def logits(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        return logits

    def forward(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        return logits


class BinaryIntermediateBranch(IntermediateBranch):
    def __init__(self, classifier: nn.Module,
                 # binary_classifier: nn.Module = None,
                 # constant_binary_output=None,
                 preprocessing: nn.Module = None,
                 return_one=False):

        super().__init__(classifier, preprocessing)

        # self.c1 = deepcopy(self.classifier)
        # self.c1.add_module('bin', nn.Linear(10, 1))
        # self.c1.add_module('s', nn.Sigmoid())
        self.return_one = return_one

        # if binary_classifier is None and constant_binary_output is None:
        #     assert False

        # if binary_classifier is None:
        #     if not isinstance(constant_binary_output, (float, int)):
        #         assert False
        #
        #     binary_classifier = lambda x: torch.full((x.shape[0], 1),
        #                                              constant_binary_output,
        #                                              device=x.device)
        #
        # self.binary_classifier = binary_classifier

    def logits(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        if not self.return_one:
            logits = logits[:, :-1]

        return logits

    def forward(self, x):
        embs = self.preprocess(x)
        logits = self.classifier(embs)

        if not self.return_one:
            logits, bin = logits[:, :-1], logits[:, -1:]
            bin = torch.sigmoid(bin)
        else:
            bin = torch.ones((x.shape[0], 1), device=x.device)

        # if self.return_one:
        #     bin = torch.ones_like(bin)
        # bin = self.c1(embs)
        # else:
        # bin = self.binary_classifier(logits)

        return logits, bin


def conv_cost(image_shape, m):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
    bias_ops = 1 if m.bias is not None else 0

    curr_im_size = (image_shape[1] / m.stride[0], image_shape[2] / m.stride[1])

    # nelement = reduce(mul, curr_im_size, 1)
    #
    # total_ops = nelement * (m.in_channels // m.groups * kernel_ops + bias_ops)

    # curr_im_size = (hparams['im_size'][0] / m.stride[0], hparams['im_size'][1] / m.stride[1])
    cost = m.kernel_size[0] * \
           m.kernel_size[1] * \
           m.in_channels * \
           m.out_channels * \
           curr_im_size[0] * \
           curr_im_size[1]

    return cost


def maxpool_cost(image_shape, m):
    curr_im_size = reduce(mul, image_shape, 1)
    cost = image_shape[1] * image_shape[2] * image_shape[0]
    # hparams['im_size'] = (hparams['im_size'][0]/m.kernel_size, hparams['im_size'][1]/m.kernel_size)
    return cost


# avgpool_cost = maxpool_cost  # To check
def avgpool_cost(image_shape, m):
    c = reduce(mul, image_shape, 1)
    return 0


def dense_cost(image_shape, m):
    cost = m.in_features * m.out_features
    return cost


def sequential_cost(input_sample, m):
    cost = 0
    for name, m_int in m.named_children():
        image_shape = input_sample.shape[1:]
        c = module_cost(input_sample, m_int)
        input_sample = m_int(input_sample)
        cost += c
    return cost


def module_cost(input_sample, m):
    image_shape = tuple(input_sample.shape[1:])

    if isinstance(m, nn.Conv2d):
        cost = conv_cost(image_shape, m)
    elif isinstance(m, nn.MaxPool2d):
        cost = maxpool_cost(image_shape, m)
    elif isinstance(m, nn.AvgPool2d):
        cost = avgpool_cost(image_shape, m)
    elif isinstance(m, nn.Linear):
        cost = dense_cost(image_shape, m)
    elif isinstance(m, nn.Sequential):  # == ['Sequential', 'BasicBlock']:
        cost = sequential_cost(input_sample, m)
    else:
        cost = 0

    return cost


def branches_predictions(model, predictors, sample_image=None):
    costs, shapes = model.computational_cost(sample_image)

    final = costs['final']

    # print(costs)
    # _costs = {k: v/final for k, v in costs.items()}
    # print(_costs)
    # trainable_parameters = lambda model: sum([p.numel() for p in model.parameters() if p.requires_grad == True])
    # print(trainable_parameters(model))

    # for i in range(model.n_branches()):
    #
    #     pc = module_cost(shapes[i], predictors[i])
    #     # print(i, pc)
    #     costs[i] += pc

    costs = {k: v / final for k, v in costs.items()}

    # print(costs)
    # input()

    return costs


class BranchModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def n_branches(self):
        raise NotImplemented

    @abstractmethod
    def computational_cost(self, image_shape):
        raise NotImplemented

# def main(args):
#     from models.alexnet import AlexNet
#
#     model = AlexNet()
#     print(args.input_size)
#     total_ops, total_params = profile(model, args.input_size)
#
#     for m in model.modules():
#         if len(list(m.children())) > 0: continue
#         print(m, m.total_ops)
#         # total_ops += m.total_ops
#         # total_params += m.total_params
#
#     print("#Ops: %f GOps"%(total_ops/1e9))
#     print("#Parameters: %f M"%(total_params/1e6))
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="pytorch model profiler")
#     # parser.add_argument("model", help="model to profile")
#     parser.add_argument("input_size", nargs='+', type=int, default=64,
#                         help="input size to the network")
#     args = parser.parse_args()
#     main(args)
