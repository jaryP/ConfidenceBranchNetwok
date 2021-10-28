from models.alexnet import AlexnetClassifier, AlexNet
from models.base import IntermediateBranch, BinaryIntermediateBranch

import torch
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop

from models.resnet import resnet20
from models.vgg import vgg11


def get_device(model: nn.Module):
    return next(model.parameters()).device


def get_intermediate_classifiers(model,
                                 image_size,
                                 num_classes,
                                 binary_branch=False,
                                 fix_last_layer=False):
    predictors = nn.ModuleList()
    x = torch.randn((1,) + image_size)
    outputs = model(x)

    for i, o in enumerate(outputs):
        chs = o.shape[1]

        if i == (len(outputs) - 1):
            od = torch.flatten(o, 1).shape[-1]

            if binary_branch:
                if fix_last_layer:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od, num_classes)])

                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 # constant_binary_output=1.0,
                                                 return_one=True)
                else:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od,
                                                              num_classes + 1)])

                    bias = linear_layers[-1].bias.data
                    bias[-1] = 10
                    # linear_layers[-1].bias.data = bias
                    # binary_layers = nn.Sequential(*[nn.ReLU(),
                    #                                 nn.Linear(num_classes,
                    #                                           num_classes),
                    #                                 nn.ReLU(),
                    #                                 nn.Linear(num_classes, 1),
                    #                                 nn.Sigmoid()
                    #                                 ])
                    #
                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 # binary_classifier=binary_layers
                                                 )
            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes)])

                b = IntermediateBranch(preprocessing=nn.Flatten(),
                                       classifier=linear_layers)

            predictors.append(b)
        else:

            if o.shape[-1] >= 6:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=3, stride=1),
                    nn.MaxPool2d(3),
                    nn.ReLU())
            else:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=2, stride=1),
                    # nn.MaxPool2d(2),
                    nn.ReLU())
            # if i == 0:
            #     seq = nn.Sequential(
            #         nn.ReLU(),
            #         nn.Conv2d(chs, 128,
            #                   kernel_size=3),
            #         nn.MaxPool2d(3),
            #         # nn.ReLU(),
            #         # nn.Conv2d(chs * 2, chs * 2,
            #         #           kernel_size=3),
            #         nn.ReLU())
            # else:
            #     seq = nn.Sequential(nn.MaxPool2d(3),
            #                         nn.ReLU(),
            #                         nn.Conv2d(chs, 128, kernel_size=3),
            #                         nn.ReLU())

            seq.add_module('flatten', nn.Flatten())

            output = seq(o)
            output = torch.flatten(output, 1)
            od = output.shape[-1]

            if binary_branch:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes + 1)])
                bias = linear_layers[-1].bias.data
                bias[-1] = 10
                # linear_layers[-1].bias.data = bias
                # binary_layers = nn.Sequential(*[nn.ReLU(),
                #                                 nn.Linear(num_classes, num_classes),
                #                                 nn.ReLU(),
                #                                 nn.Linear(num_classes, 1),
                #                                 nn.Sigmoid()
                #                                 ])

                predictors.append(
                    BinaryIntermediateBranch(preprocessing=seq,
                                             classifier=linear_layers,
                                             # binary_classifier=binary_layers
                                             ))

            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes)])

                predictors.append(IntermediateBranch(preprocessing=seq,
                                                     classifier=linear_layers))
            predictors[-1](o)

    return predictors


def get_model(name, image_size, classes, get_binaries=False,
              fix_last_layer=False):
    name = name.lower()
    if name == 'alexnet':
        model = AlexNet(image_size[0])
    elif 'vgg' in name:
        if name == 'vgg11':
            model = vgg11()
        else:
            assert False
    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20(None)
        else:
            assert False
    else:
        assert False

    classifiers = get_intermediate_classifiers(model,
                                               image_size,
                                               classes,
                                               binary_branch=get_binaries,
                                               fix_last_layer=fix_last_layer)

    return model, classifiers


def get_dataset(name, model_name, augmentation=False):
    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'flat_mnist':
        t = Compose([ToTensor(),
                     Normalize(
                         (0.1307,), (0.3081,)),
                     torch.nn.Flatten(0)
                     ])

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 28 * 28

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   Normalize((0.4376821, 0.4437697, 0.47280442),
                             (0.19803012, 0.20101562, 0.19703614))])

        t = [
            ToTensor(),
            Normalize((0.4376821, 0.4437697, 0.47280442),
                      (0.19803012, 0.20101562, 0.19703614))]

        # if 'resnet' in model_name:
        #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
        #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.SVHN(
            root='~/loss_landscape_dataset/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/loss_landscape_dataset/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

        t = [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar100':
        if augmentation:
            tt = [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(), ]
        else:
            tt = []

        tt.extend([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))])

        t = [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 100

    # elif name == 'tinyimagenet':
    #     tt = [
    #         ToTensor(),
    #         # transforms.RandomCrop(56),
    #         RandomResizedCrop(64),
    #         RandomHorizontalFlip(),
    #         Normalize((0.4802, 0.4481, 0.3975),
    #                              (0.2302, 0.2265, 0.2262))
    #     ]
    #
    #     t = [
    #         ToTensor(),
    #         Normalize((0.4802, 0.4481, 0.3975),
    #                              (0.2302, 0.2265, 0.2262))
    #     ]
    #
    #     transform = Compose(t)
    #     train_transform = Compose(tt)
    #
    #     # train_set = TinyImageNet(
    #     #     root='~/loss_landscape_dataset/tiny-imagenet-200', split='train',
    #     #     transform=transform)
    #
    #     train_set = datasets.ImageFolder('~/loss_landscape_dataset/tiny-imagenet-200/train',
    #                                      transform=train_transform)
    #
    #     # for x, y in train_set:
    #     #     if x.shape[exp_0] == 1:
    #     #         print(x.shape[exp_0] == 1)
    #
    #     # test_set = TinyImageNet(
    #     #     root='~/loss_landscape_dataset/tiny-imagenet-200', split='val',
    #     #     transform=train_transform)
    #     test_set = datasets.ImageFolder('~/loss_landscape_dataset/tiny-imagenet-200/val',
    #                                     transform=transform)
    #
    #     # for x, y in test_set:
    #     #     if x.shape[exp_0] == 1:
    #     #         print(x.shape[exp_0] == 1)
    #
    #     input_size, classes = 3, 200

    else:
        assert False

    return train_set, test_set, input_size, classes


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0):
    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam or sgd')
