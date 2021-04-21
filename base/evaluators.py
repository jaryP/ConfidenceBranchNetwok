from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch import nn

from models.base import BranchModel


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    expected, predicted = np.asarray(expected), np.asarray(predicted)
    assert len(expected) == len(predicted)
    assert len(predicted.shape) == 2 and len(expected.shape) == 1
    assert predicted.shape[1] >= max(topk)

    res = defaultdict(int)
    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


def standard_eval(model: BranchModel, dataset_loader,
                  device='cpu', topk=None):
    true_labels = []
    pred_labels = []

    model.eval()

    with torch.no_grad():
        for x, y in dataset_loader:

            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            true_labels.extend(y.tolist())

            top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
            pred_labels.extend(top_classes.tolist())

            # for i in range(len(preds) - 1):
            #     p = predictors[i](preds[i])
            #     # pred = torch.argmax(pred, -1)
            #     top_classes = torch.topk(p, p.size(-1))[1]
            #     pred_labels[i].extend(top_classes.tolist())

        scores = accuracy_score(np.asarray(true_labels),
                                np.asarray(pred_labels), topk=topk)

    return scores


def branches_eval(model: BranchModel, predictors, dataset_loader,
                  device='cpu', topk=None):
    true_labels = []
    pred_labels = defaultdict(list)

    model.eval()
    predictors.eval()

    with torch.no_grad():
        for x, y in dataset_loader:
            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            true_labels.extend(y.tolist())

            top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
            pred_labels['final'].extend(top_classes.tolist())

            for i in range(len(preds) - 1):
                p = predictors[i](preds[i])
                # pred = torch.argmax(pred, -1)
                top_classes = torch.topk(p, p.size(-1))[1]
                pred_labels[i].extend(top_classes.tolist())

        true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels, np.asarray(p), topk=topk)
                  for i, p in pred_labels.items()}

    return scores


def branches_entropy(model: BranchModel,
                     predictors: nn.ModuleList,
                     threshold: Union[List[float], float],
                     dataset_loader,
                     device='cpu', topk=None):

    model.eval()
    predictors.eval()

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    with torch.no_grad():
        for x, y in dataset_loader:
            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            for bi in range(x.shape[0]):
                found = False

                for i, predictor in enumerate(predictors):
                    logits = preds[i][bi]
                    p = predictor(logits.unsqueeze(0)).squeeze(0)
                    sf = nn.functional.softmax(p, -1)
                    h = -sf.log() * sf
                    # print(bi, i, h.sum())
                    h = h / np.log(sf.shape[-1])
                    h = h.sum()
                    if h < threshold[i]:
                        top_classes = torch.topk(p, p.size(-1))[1]
                        pred_labels[i].append(top_classes.tolist())
                        true_labels[i].append(y[bi].item())
                        exits_counter[i] += 1
                        found = True
                        break

                if not found:
                    top_classes = torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    true_labels['final'].append(y[bi].item())
                    pred_labels['final'].append(top_classes.tolist())
                    exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate([true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        scores['branches_counter'] = exits_counter

    return scores



def branches_binary(model: BranchModel,
                     predictors: nn.ModuleList,
                     binary_classifiers: nn.ModuleList,
                     dataset_loader,
                     threshold: Union[List[float], float]=None,
                     device='cpu', topk=None):

    model.eval()
    predictors.eval()

    if threshold is None:
        threshold = 0.5

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    with torch.no_grad():
        for x, y in dataset_loader:
            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            for bi in range(x.shape[0]):
                found = False

                for i, predictor in enumerate(binary_classifiers):
                    logits = preds[i][bi]
                    h = predictor(logits.unsqueeze(0)).squeeze(0)
                    if h >= threshold[i]:
                        p = predictors[i](logits.unsqueeze(0)).squeeze(0)
                        top_classes = torch.topk(p, p.size(-1))[1]
                        pred_labels[i].append(top_classes.tolist())
                        true_labels[i].append(y[bi].item())
                        exits_counter[i] += 1
                        found = True
                        break

                if not found:
                    top_classes = torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    true_labels['final'].append(y[bi].item())
                    pred_labels['final'].append(top_classes.tolist())
                    exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate([true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        scores['branches_counter'] = exits_counter

    return scores
