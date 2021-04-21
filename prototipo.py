from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam


class branch_lenet(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=10):
        super().__init__()
        self.f1 = nn.Linear(input_size, input_size // 2)
        self.b1 = nn.Linear(input_size // 2, output_size)

        self.f2 = nn.Linear(input_size // 2, input_size // 2)
        self.b2 = nn.Linear(input_size // 2, output_size)

        self.f3 = nn.Linear(input_size // 2, input_size // 2)
        self.b3 = nn.Linear(input_size // 2, output_size)

        self.f4 = nn.Linear(input_size // 2, output_size)
        self.branches = 3

    def forward(self, x):
        branchs = []

        for i in [1, 2, 3]:
            f = getattr(self, 'f{}'.format(i))
            b = getattr(self, 'b{}'.format(i))

            x = f(x)
            x = torch.relu(x)

            b = b(x)
            branchs.append(b)

        branchs.append(self.f4(x))
        return branchs


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    assert len(expected) == len(predicted)
    assert predicted.shape[1] >= max(topk)

    res = defaultdict(int)
    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


def joint_training():
    EPOCHS = 10
    DEVICE = 'cuda'

    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)),
                                        torch.nn.Flatten(0)
                                        ])

    train_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=True,
        transform=t,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=False,
        transform=t,
        download=True
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False)

    classes = 10
    input_size = 28 * 28
    model = branch_lenet()

    opt = Adam(model.parameters(), lr=0.001)

    w = [1 / model.branches] * model.branches + [1]
    model.to(DEVICE)
    for e in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = 0
            for i, p in enumerate(preds):
                l = torch.nn.functional.cross_entropy(p, y, reduction='mean')
                loss += l * w[i]

            opt.zero_grad()
            loss.backward()
            opt.step()

        true_labels = []
        pred_labels = defaultdict(list)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                print(i, scores[1])


def random_training():
    EPOCHS = 10
    DEVICE = 'cuda'

    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)),
                                        torch.nn.Flatten(0)
                                        ])

    train_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=True,
        transform=t,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=False,
        transform=t,
        download=True
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False)

    classes = 10
    input_size = 28 * 28
    model = branch_lenet()

    opt = Adam(model.parameters(), lr=0.001)

    w = [1 / model.branches] * model.branches + [1]
    model.to(DEVICE)
    for e in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)

            p = np.random.choice(preds, 1)[0]
            loss = torch.nn.functional.cross_entropy(p, y, reduction='mean')

            opt.zero_grad()
            loss.backward()
            opt.step()

        true_labels = []
        pred_labels = defaultdict(list)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                print(i, scores[1])


def thompson_training(random=False):
    EPOCHS = 10
    EVAL_PERCENTAGE = 0.1
    DEVICE = 'cuda'
    DECAY = 0.9

    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)),
                                        torch.nn.Flatten(0)
                                        ])

    train_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=True,
        transform=t,
        download=True
    )

    test_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=False,
        transform=t,
        download=True
    )

    train_len = len(train_set)
    eval_len = int(train_len * EVAL_PERCENTAGE)
    train_len = train_len - eval_len

    train_set, eval_set = torch.utils.data.random_split(train_set,
                                                        [train_len, eval_len])

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False)

    eval_loader = torch.utils.data.DataLoader(dataset=eval_set,
                                              batch_size=32,
                                              shuffle=False)

    classes = 10
    input_size = 28 * 28
    model = branch_lenet()

    opt = Adam(model.parameters(), lr=0.001)

    prior_a = [1 / model.branches] * model.branches + [1]
    prior_b = [1 / model.branches] * model.branches + [1]

    posterior_a = deepcopy(prior_a)
    posterior_b = deepcopy(prior_b)

    picks = []

    model.to(DEVICE)
    t = np.linspace(10, 0.5, EPOCHS)

    for e in range(EPOCHS):

        print(posterior_a, posterior_b)

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)

            if e > 0:
                samples = [np.random.beta(a, b) for a, b in
                           zip(posterior_a, posterior_b)]
                if random:
                    samples = np.asarray(samples)
                    samples = np.exp(samples) / np.exp(samples).sum()
                    # input(samples)
                    # samples /= np.sum(samples)
                    samples /= t[e]
                    # samples = np.exp(samples) / np.exp(samples).sum()
                    samples = samples / samples.sum()
                    # print(t[e])
                    # input(samples)
                    i = np.random.choice(range(len(prior_a)), p=samples)
                else:
                    i = np.argmax(samples)
            else:
                i = np.random.randint(0, len(prior_a))

            p = preds[i]

            loss = torch.nn.functional.cross_entropy(p, y, reduction='mean')

            opt.zero_grad()
            loss.backward()
            opt.step()

        true_labels = []
        pred_labels = defaultdict(list)

        print([a / (a + b) for a, b in zip(posterior_a, posterior_b)])

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                print(i, scores[1])

        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                acc = scores[1]
                # prior_b[i] *= DECAY + 1 - acc
                # prior_a[i] *= DECAY + acc

                # posterior_b[i] = 1 - acc
                # posterior_a[i] = acc
                posterior_b[i] *= DECAY + 1 - acc
                posterior_a[i] *= DECAY + acc


def beta_training():
    from scipy.stats import beta

    EPOCHS = 10
    EVAL_PERCENTAGE = 0.1
    C = 1
    DEVICE = 'cuda'

    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,)),
                                        torch.nn.Flatten(0)
                                        ])

    train_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=True,
        transform=t,
        download=True
    )

    train_len = len(train_set)
    eval_len = int(train_len * EVAL_PERCENTAGE)
    train_len = train_len - eval_len

    train_set, eval_set = torch.utils.data.random_split(train_set,
                                                        [train_len, eval_len])

    test_set = torchvision.datasets.MNIST(
        root='./datasets/mnist/',
        train=False,
        transform=t,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False)

    eval_loader = torch.utils.data.DataLoader(dataset=eval_set,
                                              batch_size=32,
                                              shuffle=False)

    classes = 10
    input_size = 28 * 28
    model = branch_lenet()

    opt = Adam(model.parameters(), lr=0.001)

    prior_a = [1 / model.branches] * model.branches + [1]
    prior_b = [1 / model.branches] * model.branches + [1]
    # prior_b = [1] * (model.branches + 1)

    model.to(DEVICE)

    for e in range(EPOCHS):
        print(prior_a, prior_b)

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)

            # samples = [np.random.beta(a, b) for a, b in
            #            zip(prior_a, prior_b)]

            samples = [a / float(a + b) + beta.std(
                a, b) * C for a, b in
                       zip(prior_a, prior_b)]

            i = np.argmax(samples)
            p = preds[i]

            loss = torch.nn.functional.cross_entropy(p, y, reduction='mean')

            opt.zero_grad()
            loss.backward()
            opt.step()

        true_labels = []
        pred_labels = defaultdict(list)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                print(i, scores[1])

        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)

                true_labels.extend(y.tolist())
                for i in range(len(preds)):
                    pred = preds[i]
                    # pred = torch.argmax(pred, -1)
                    top_classes = torch.topk(pred, pred.size(-1))[1]
                    pred_labels[i].extend(top_classes.tolist())

            for i, p in pred_labels.items():
                scores = accuracy_score(np.asarray(true_labels),
                                        np.asarray(p))
                acc = scores[1]
                prior_b[i] += 1 - acc
                prior_a[i] += acc

                print(i, scores[1])


# print('Joint training')
# joint_training()
# print('Random training')
# random_training()
# print('Thompson training')
# thompson_training(True)
# print('BETA training')
# beta_training()
