from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from bayesian.posteriors import BayesianPosterior, BayesianHead, BayesianHeads
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

    with torch.no_grad():
        for x, y in dataset_loader:
            model.eval()
            predictors.eval()

            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            true_labels.extend(y.tolist())

            top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
            pred_labels['final'].extend(top_classes.tolist())

            for i in range(len(preds)):
                p = predictors[i](preds[i])
                # pred = torch.argmax(pred, -1)
                top_classes = torch.topk(p, p.size(-1))[1]
                pred_labels[i].extend(top_classes.tolist())

        true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels, np.asarray(p), topk=topk)
                  for i, p in pred_labels.items()}

    return scores


def branches_entropy(model: BranchModel,
                     predictors: Union[nn.ModuleList, BayesianHeads],
                     threshold: Union[List[float], float],
                     dataset_loader,
                     device='cpu',
                     samples=1,
                     topk=None):
    model.eval()
    predictors.eval()

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    with torch.no_grad():
        for x, y in tqdm(dataset_loader):
            x, y = x.to(device), y.to(device)
            final_preds, preds = model(x)

            logits = []

            for i, predictor in enumerate(predictors):
                if isinstance(predictor, BayesianHead):
                    p = predictor(logits=preds[i],
                                  branch_index=i, samples=samples)
                else:
                    p = predictor(preds[i])

                logits.append(p)

            for bi in range(x.shape[0]):
                found = False

                for i, predictor in enumerate(predictors):
                    p = logits[i][bi]  # .unsqueeze(0)
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
                    top_classes = \
                        torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    true_labels['final'].append(y[bi].item())
                    pred_labels['final'].append(top_classes.tolist())
                    exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate(
            [true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        # scores['branches_counter'] = exits_counter

    return scores, exits_counter


@torch.no_grad()
def eval_branches_entropy(model: BranchModel,
                          predictors: Union[nn.ModuleList, BayesianHeads],
                          percentile,
                          # threshold: Union[List[float], float],
                          dataset_loader,
                          eval_loader,
                          device='cpu',
                          samples=1,
                          topk=None):
    model.eval()
    predictors.eval()

    # if isinstance(threshold, float):
    #     threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    h = defaultdict(list)
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        final_preds, preds = model(x)

        for bi in range(x.shape[0]):
            found = False
            logits = []

            for i, predictor in enumerate(predictors):
                if isinstance(predictor, BayesianHead):
                    p = predictor(logits=preds[i],
                                  branch_index=i, samples=samples)
                else:
                    p = predictor(preds[i])

                logits.append(p)

            for i, predictor in enumerate(predictors):
                p = logits[i][bi]  # .unsqueeze(0)

                if torch.argmax(p) == y[bi]:
                    sf = nn.functional.softmax(p, -1)
                    _h = -sf.log() * sf
                    # print(bi, i, h.sum())
                    _h = _h / np.log(sf.shape[-1])
                    _h = _h.sum()
                    h[i].append(_h.item ())

    # print(h)
    threshold = [np.quantile(h[i], percentile) for i in range(len(h))]
    # print(threshold)
    # for k, hs in h.items():
    #     print(k, np.quantile(hs, 0.5))
    # return

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        final_preds, preds = model(x)

        logits = []

        for i, predictor in enumerate(predictors):
            if isinstance(predictor, BayesianHead):
                p = predictor(logits=preds[i],
                              branch_index=i, samples=samples)
            else:
                p = predictor(preds[i])

            logits.append(p)

        for bi in range(x.shape[0]):
            found = False

            for i, predictor in enumerate(predictors):
                p = logits[i][bi]  # .unsqueeze(0)
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
                top_classes = \
                    torch.topk(final_preds[bi], final_preds.size(-1))[1]
                true_labels['final'].append(y[bi].item())
                pred_labels['final'].append(top_classes.tolist())
                exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate(
            [true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        # scores['branches_counter'] = exits_counter

    return scores, exits_counter


def branches_binary(model: BranchModel,
                    predictors: nn.ModuleList,
                    binary_classifiers: Union[nn.ModuleList, BayesianPosterior],
                    dataset_loader,
                    threshold: Union[List[float], float] = None,
                    device='cpu', samples=1, topk=None):
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

                for i in range(len(binary_classifiers)):
                    logits = preds[i][bi].unsqueeze(0)

                    if isinstance(binary_classifiers, BayesianPosterior):
                        h = binary_classifiers(logits=logits,
                                               samples=samples,
                                               branch_index=i).squeeze(0)
                    else:
                        binary_predictor = binary_classifiers[i]
                        h = binary_predictor(logits).squeeze(0)

                    if bi == 0:
                        print(i, h, threshold)

                    if h >= threshold[i]:
                        p = predictors[i](logits).squeeze(0)
                        top_classes = torch.topk(p, p.size(-1))[1]
                        pred_labels[i].append(top_classes.tolist())
                        true_labels[i].append(y[bi].item())
                        exits_counter[i] += 1
                        found = True
                        break

                if not found:
                    top_classes = \
                        torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    true_labels['final'].append(y[bi].item())
                    pred_labels['final'].append(top_classes.tolist())
                    exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate(
            [true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        # scores['branches_counter'] = exits_counter

    return scores, exits_counter


def branches_mean(model: BranchModel,
                  predictors: nn.ModuleList,
                  posteriors: BayesianPosterior,
                  dataset_loader,
                  c=1,
                  samples=1,
                  threshold: Union[List[float], float] = None,
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

            ps = [posteriors.get_posterior(branch_index=j, logits=bo)
                  for j, bo in enumerate(preds)]

            for bi in range(x.shape[0]):
                found = False

                for i in range(len(posteriors)):
                    p = ps[i]
                    mean, var = p.mean[bi], p.variance[bi]
                    th = mean + c * var

                    logits = preds[i][bi].unsqueeze(0)

                    h = posteriors(logits=logits,
                                   branch_index=i,
                                   samples=samples).squeeze(0)

                    if h >= th:
                        p = predictors[i](logits).squeeze(0)
                        top_classes = torch.topk(p, p.size(-1))[1]
                        pred_labels[i].append(top_classes.tolist())
                        true_labels[i].append(y[bi].item())
                        exits_counter[i] += 1
                        found = True
                        break

                if not found:
                    top_classes = \
                        torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    true_labels['final'].append(y[bi].item())
                    pred_labels['final'].append(top_classes.tolist())
                    exits_counter['final'] += 1

        # true_labels = np.asarray(true_labels)
        scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                  for i, p in pred_labels.items()}

        all_labels = np.concatenate(
            [true_labels[i] for i, p in pred_labels.items()])
        all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

        scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

        # scores['branches_counter'] = exits_counter

    return scores, exits_counter
