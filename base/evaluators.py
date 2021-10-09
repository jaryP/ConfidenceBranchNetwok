from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet
from tqdm import tqdm

# from bayesian.posteriors import BayesianPosterior, BayesianHead, \
#     BayesianPredictors
from models.base import BranchModel, branches_predictions, IntermediateBranch
from utils import get_device


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    expected, predicted = np.asarray(expected), np.asarray(predicted)
    assert len(expected) == len(predicted)
    assert len(predicted.shape) == 2 and len(expected.shape) == 1
    assert predicted.shape[1] >= max(topk)

    res = {k: 0 for k in topk}

    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


@torch.no_grad()
def standard_eval(model: BranchModel,
                  classifier: IntermediateBranch,
                  dataset_loader):
    # true_labels = []
    # pred_labels = []
    device = get_device(model)

    model.eval()

    total = 0
    correct = 0

    # with torch.no_grad():
    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        final_preds = classifier.logits(model(x)[-1])
        # true_labels.extend(y.tolist())
        pred = torch.argmax(final_preds, 1)

        total += y.size(0)
        correct += (pred == y).sum().item()

        # top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
        # pred_labels.extend(top_classes.tolist())

        # for i in range(len(preds) - 1):
        #     p = predictors[i](preds[i])
        #     # pred = torch.argmax(pred, -1)
        #     top_classes = torch.topk(p, p.size(-1))[1]
        #     pred_labels[i].extend(top_classes.tolist())

    score = correct / total

    # scores = accuracy_score(np.asarray(true_labels),
    #                         np.asarray(pred_labels), topk=topk)

    return score


@torch.no_grad()
def branches_eval(model: BranchModel, predictors, dataset_loader):
    # true_labels = []
    # pred_labels = defaultdict(list)
    device = get_device(model)

    model.eval()
    predictors.eval()

    scores = {}

    for i in range(len(predictors)):
        predictor = predictors[i]

        total = 0
        correct = 0

        for x, y in dataset_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)[i]

            pred = predictor.logits(pred)
            # true_labels.extend(y.tolist())
            pred = torch.argmax(pred, 1)

            total += y.size(0)
            correct += (pred == y).sum().item()

        if i == (len(predictors) - 1):
            i = 'final'

        scores[i] = correct / total

    return scores


@torch.no_grad()
def entropy_eval(model: BranchModel,
                 predictors: nn.ModuleList,
                 threshold: Union[List[float], float],
                 dataset_loader):
    model.eval()
    predictors.eval()
    device = get_device(model)

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    # exits_counter = defaultdict(int)
    #
    # true_labels = defaultdict(list)
    # pred_labels = defaultdict(list)
    #
    exits_counter = defaultdict(int)
    exits_corrected = defaultdict(int)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []

        for j, bo in enumerate(preds):
            l = predictors[j].logits(bo)
            # distributions.append(b)
            logits.append(l)

        # distributions = torch.stack(distributions, 0)
        logits = torch.stack(logits, 0)

        for bi in range(x.shape[0]):
            found = False

            for i, predictor in enumerate(predictors):
                p = logits[i][bi]  # .unsqueeze(0)
                sf = nn.functional.softmax(p, -1)
                h = -(sf + 1e-12).log() * sf
                # print(bi, i, h.sum())
                h = h / np.log(sf.shape[-1])
                h = h.sum()

                if h < threshold[i]:
                    pred = torch.argmax(p)

                    if pred == y[bi]:
                        exits_corrected[i] += 1

                    # total += y.size(0)
                    # correct += (pred == y).sum().item()
                    #
                    # top_classes = torch.topk(p, p.size(-1))[1]
                    #
                    # pred_labels[i].append(top_classes.tolist())
                    # true_labels[i].append(y[bi].item())
                    #
                    # exits_counter[i] += 1
                    # top_classes = torch.topk(p, p.size(-1))[1]
                    # pred_labels[i].append(top_classes.tolist())
                    # true_labels[i].append(y[bi].item())
                    exits_counter[i] += 1
                    found = True
                    break

            if not found:
                i = len(predictors) - 1
                p = logits[i][bi]

                exits_counter[i] += 1
                pred = torch.argmax(p)

                if pred == y[bi]:
                    exits_corrected[i] += 1
                # exits_counter[i] += 1

                # top_classes = \
                #     torch.topk(final_preds[bi], final_preds.size(-1))[1]
                # true_labels['final'].append(y[bi].item())
                # pred_labels['final'].append(top_classes.tolist())
                # exits_counter['final'] += 1

    branches_scores = {}
    tot = 0
    correctly_predicted = 0

    # print(exits_corrected)
    # print(exits_counter)

    for k in exits_corrected:
        correct = exits_corrected[k]
        counter = exits_counter.get(k, 0)

        if counter == 0:
            score = 0
        else:
            score = correct / counter

        branches_scores[k] = score

        tot += counter
        correctly_predicted += correct

    # all_labels = np.concatenate(
    #     [true_labels[i] for i, p in pred_labels.items() if len(p) > 0])
    # all_preds = np.concatenate(
    #     [p for i, p in pred_labels.items() if len(p) > 0], 0)

    branches_scores['global'] = correctly_predicted / tot

    # scores['branches_counter'] = exits_counter

    return branches_scores, exits_counter


@torch.no_grad()
def binary_eval(model: BranchModel,
                predictors: nn.ModuleList,
                dataset_loader,
                epsilon: Union[List[float], float] = None,
                cumulative_threshold=False):
    model.eval()
    predictors.eval()
    # binary_classifiers.eval()
    device = get_device(model)

    if epsilon is None:
        epsilon = 0.5

    if isinstance(epsilon, float):
        epsilon = [epsilon] * model.n_branches()

    exits_counter = defaultdict(int)
    exits_corrected = defaultdict(int)

    # exit_correct = defaultdict(int)

    # true_labels = defaultdict(list)
    # pred_labels = defaultdict(list)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []

        for j, bo in enumerate(preds):
            l, b = predictors[j](bo)
            distributions.append(b)
            logits.append(l)

        distributions = torch.stack(distributions, 0)
        logits = torch.stack(logits, 0)

        # print(distributions.shape, logits.shape, x.shape)

        # for d in range(len(distributions)):
        #     logits = preds[i]
        #
        #     if isinstance(binary_classifiers, BayesianPosterior):
        #         h = binary_classifiers(logits=logits,
        #                                samples=samples,
        #                                branch_index=i)
        #     else:
        #         binary_predictor = binary_classifiers[i]
        #         h = binary_predictor(logits)
        #
        #     predictions.append(predictors[i](logits))
        #     hs.append(h)

        # if cumulative_threshold:
        #     epsilon = 1 - epsilon

        for bi in range(x.shape[0]):
            threshold = 1
            found = False
            ps = []

            for i in range(logits.shape[0]):

                b = distributions[i][bi]
                ps.append(b.item())

                if cumulative_threshold:
                    if i > 0:
                        _p = np.asarray(ps)
                        _p[1:] *= np.cumprod(1 - _p[:-1])
                        # _p = np.cumprod(1 - np.asarray(ps)[:-1])
                        # print(_p, _p.sum(), b, b * _p[-1])
                        b = _p.sum()
                        # b = b * _p[-1]
                    # else:
                    #     b = 1 - b

                if b >= epsilon[i]:
                    p = logits[i][bi]
                    # print(p.shape)
                    pred = torch.argmax(p)
                    if pred == y[bi]:
                        exits_corrected[i] += 1

                    # total += y.size(0)
                    # correct += (pred == y).sum().item()
                    #
                    # top_classes = torch.topk(p, p.size(-1))[1]
                    #
                    # pred_labels[i].append(top_classes.tolist())
                    # true_labels[i].append(y[bi].item())
                    #
                    exits_counter[i] += 1
                    #
                    #
                    #
                    # found = True
                    break

                # for i in range(len(binary_classifiers)):
                #     logits = preds[i][bi].unsqueeze(0)
                #
                #     if isinstance(binary_classifiers, BayesianPosterior):
                #         h = binary_classifiers(logits=logits,
                #                                samples=samples,
                #                                branch_index=i).squeeze(0)
                #     else:
                #         binary_predictor = binary_classifiers[i]
                #         h = binary_predictor(logits).squeeze(0)
                #
                #     if h >= threshold[i]:
                #         p = predictors[i](logits).squeeze(0)
                #         top_classes = torch.topk(p, p.size(-1))[1]
                #         pred_labels[i].append(top_classes.tolist())
                #         true_labels[i].append(y[bi].item())
                #         exits_counter[i] += 1
                #         found = True
                #         break

            # if not found:
            #     top_classes = \
            #         torch.topk(final_preds[bi], final_preds.size(-1))[1]
            #     true_labels['final'].append(y[bi].item())
            #     pred_labels['final'].append(top_classes.tolist())
            #     exits_counter['final'] += 1

    # true_labels = np.asarray(true_labels)
    # scores = {i: accuracy_score(true_labels[i], p, topk=topk)
    #           for i, p in pred_labels.items()}

    branches_scores = {}
    tot = 0
    correctly_predicted = 0

    # print(exits_corrected)
    # print(exits_counter)

    for k in exits_corrected:
        correct = exits_corrected[k]
        counter = exits_counter.get(k, 0)

        if counter == 0:
            score = 0
        else:
            score = correct / counter

        branches_scores[k] = score

        tot += counter
        correctly_predicted += correct

    # all_labels = np.concatenate(
    #     [true_labels[i] for i, p in pred_labels.items() if len(p) > 0])
    # all_preds = np.concatenate(
    #     [p for i, p in pred_labels.items() if len(p) > 0], 0)

    branches_scores['global'] = correctly_predicted / tot

    # scores['branches_counter'] = exits_counter

    return branches_scores, exits_counter

