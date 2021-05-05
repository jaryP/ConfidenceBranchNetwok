import functools
import operator
from collections import defaultdict

import torch
from torch import nn

from models.base import BranchModel, module_cost


class AlexNet(BranchModel):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.b = 5
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2)
        self.c2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        # nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # nn.MaxPool2d(kernel_size=3, stride=2)

        # self.features = nn.Sequential(
        #     # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2),
        #     # nn.Conv2d(64, 192, kernel_size=3, padding=2),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def n_branches(self):
        return self.b

    def computational_cost(self, sample_image):
        costs = defaultdict(int)
        shapes = dict()

        if sample_image is None:
            sample_image = torch.randn((1, 3, 32, 32))
            # image_shape = (3, 32, 32)

        for i in range(1, 6):
            cc = 0
            cl = getattr(self, 'c{}'.format(i))

            cost = module_cost(sample_image, cl)
            cc += cost

            sample_image = cl(sample_image)
            shapes[i - 1] = sample_image.clone()

            if i in [2, 3, 5]:
                cost = module_cost(sample_image, nn.MaxPool2d(kernel_size=2))

                sample_image = nn.functional.max_pool2d(sample_image,
                                                        kernel_size=2)
                cc += cost

            costs[i - 1] += cc
            if i > 1:
                costs[i - 1] += costs[i - 2]

        sample_image = torch.flatten(sample_image, 1)

        cost = module_cost(sample_image,  self.fc_layers)

        shapes['final'] = sample_image.clone()
        costs['final'] = costs[self.n_branches() - 1] + cost

        return dict(costs), shapes

    def forward(self, x):
        intermediate_layers = []

        for i in range(1, 6):
            cl = getattr(self, 'c{}'.format(i))
            x = cl(x)
            intermediate_layers.append(x)
            if i in [2, 3, 5]:
                x = nn.functional.max_pool2d(x, kernel_size=2)
            x = torch.relu(x)

        # x = self.c1(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=2)
        # x = torch.relu(x)
        #
        # x = self.c2(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=2)
        # x = torch.relu(x)
        #
        # x = self.c3(x)
        # intermediate_layers.append(x)
        # x = torch.relu(x)
        #
        # x = self.c4(x)
        # intermediate_layers.append(x)
        # x = torch.relu(x)
        #
        # # self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # x = self.c5(x)
        # intermediate_layers.append(x)
        # x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        # x = torch.relu(x)
        # # conv_features = self.features(x)

        flatten = torch.flatten(x, 1)
        fc = self.fc_layers(flatten)

        return fc, intermediate_layers


if __name__ == '__main__':
    model = AlexNet()
    c = model.computational_cost()
    print(c)