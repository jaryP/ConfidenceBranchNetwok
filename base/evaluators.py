from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet
from tqdm import tqdm

from bayesian.posteriors import BayesianPosterior, BayesianHead, \
    BayesianPredictors
from models.base import BranchModel, branches_predictions, IntermediateBranch


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
                  dataset_loader,
                  device='cpu', topk=None):
    # true_labels = []
    # pred_labels = []

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
def branches_eval(model: BranchModel, predictors, dataset_loader,
                  device='cpu', topk=None):
    # true_labels = []
    # pred_labels = defaultdict(list)

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

    # predictor = predictors[i]
    #
    #
    # for x, y in dataset_loader:
    #     model.eval()
    #     predictors.eval()
    #
    #     x, y = x.to(device), y.to(device)
    #     preds = model(x)
    #
    #     final_preds = preds[-1]
    #
    #     true_labels.extend(y.tolist())
    #
    #     top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
    #     pred_labels['final'].extend(top_classes.tolist())
    #
    #     for i in range(len(preds) - 1):
    #         p = predictors[i].logits(preds[i])
    #         # pred = torch.argmax(pred, -1)
    #         top_classes = torch.topk(p, p.size(-1))[1]
    #         pred_labels[i].extend(top_classes.tolist())
    #
    # true_labels = np.asarray(true_labels)
    # scores = {i: accuracy_score(true_labels, np.asarray(p), topk=topk)
    #           for i, p in pred_labels.items()}

    return scores


@torch.no_grad()
def branches_entropy(model: BranchModel,
                     predictors: Union[nn.ModuleList, BayesianPredictors],
                     threshold: Union[List[float], float],
                     dataset_loader,
                     device='cpu',
                     samples=1,
                     topk=None):
    model.eval()
    predictors.eval()

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
def eval_branches_entropy(model: BranchModel,
                          predictors: Union[nn.ModuleList, BayesianPredictors],
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
                    _h = -(sf + 1e-12).log() * sf
                    # print(bi, i, h.sum())
                    _h = _h / np.log(sf.shape[-1])
                    _h = _h.sum()
                    h[i].append(_h.item())

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
                h = -(sf + 1e-12).log() * sf
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
def binary_eval(model: BranchModel,
                predictors: nn.ModuleList,
                # binary_classifiers: Union[nn.ModuleList, BayesianPosterior],
                dataset_loader,
                epsilon: Union[List[float], float] = None,
                device='cpu',
                cumulative_threshold=False,
                samples=1, topk=None):
    model.eval()
    predictors.eval()
    # binary_classifiers.eval()

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


@torch.no_grad()
def branches_binary(model: BranchModel,
                    predictors: nn.ModuleList,
                    binary_classifiers: Union[nn.ModuleList, BayesianPosterior],
                    dataset_loader,
                    threshold: Union[List[float], float] = None,
                    device='cpu', samples=1, topk=None):
    model.eval()
    predictors.eval()
    binary_classifiers.eval()

    if threshold is None:
        threshold = 0.5

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        final_preds, preds = model(x)

        hs = []
        predictions = []

        for i in range(len(binary_classifiers)):
            logits = preds[i]

            if isinstance(binary_classifiers, BayesianPosterior):
                h = binary_classifiers(logits=logits,
                                       samples=samples,
                                       branch_index=i)
            else:
                binary_predictor = binary_classifiers[i]
                h = binary_predictor(logits)

            predictions.append(predictors[i](logits))
            hs.append(h)

        for bi in range(x.shape[0]):
            found = False
            for i in range(len(binary_classifiers)):
                if hs[i][bi] >= threshold[i]:
                    p = predictions[i][bi]
                    top_classes = torch.topk(p, p.size(-1))[1]
                    pred_labels[i].append(top_classes.tolist())
                    true_labels[i].append(y[bi].item())
                    exits_counter[i] += 1
                    found = True
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
        [true_labels[i] for i, p in pred_labels.items() if len(p) > 0])
    all_preds = np.concatenate(
        [p for i, p in pred_labels.items() if len(p) > 0], 0)

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


@torch.no_grad()
def branches_entropy_gain(model: BranchModel,
                          predictors: Union[nn.ModuleList, BayesianPredictors],
                          threshold: Union[List[float], float],
                          dataset_loader,
                          device='cpu',
                          samples=1,
                          topk=None):
    sample_image = next(iter(dataset_loader))[0][1:2].to(device)
    costs = branches_predictions(model, predictors, sample_image)

    model.eval()
    predictors.eval()

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)

    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)
    pred_labels_final = defaultdict(list)

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
                h = -(sf + 1e-12).log() * sf
                # print(bi, i, h.sum())
                h = h / np.log(sf.shape[-1])
                h = h.sum()
                if h < threshold[i]:
                    top_classes = torch.topk(p, p.size(-1))[1]
                    pred_labels[i].append(top_classes.tolist())

                    top_classes = \
                        torch.topk(final_preds[bi], final_preds.size(-1))[1]
                    pred_labels_final[i].append(top_classes.tolist())

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

    final_scores = {i: accuracy_score(true_labels[i], p, topk=topk)
                    for i, p in pred_labels_final.items()}

    # input()

    keys = list(range(model.n_branches())) + ['final']
    all_labels = []
    all_preds = []

    for k in keys:
        if len(true_labels[k]) == 0:
            continue
        all_labels.append(true_labels[k])
        all_preds.append(pred_labels[k])

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds, 0)

    # all_labels = np.concatenate(
    #     [true_labels[i] for i, p in pred_labels.items()])
    # all_preds = np.concatenate([p for i, p in pred_labels.items()], 0)

    print(scores)
    print(final_scores)

    c = 0
    for k, v in scores.items():
        if k == 'final':
            continue
        si = v.get(1, 0)

        p = len(true_labels[k]) / len(all_labels)
        co = costs[k]
        sf = final_scores[k].get(1, 0)

        c += p * ((si - sf) + 1 / co)
        print(p, si, sf, co, p * ((si - sf) + 1 / co))
        print()

    print(c)

    scores['global'] = accuracy_score(all_labels, all_preds, topk=topk)

    # scores['branches_counter'] = exits_counter

    return scores, exits_counter


@torch.no_grad()
def dirichlet_entropy(model: BranchModel,
                      predictors: Union[nn.ModuleList, BayesianPredictors],
                      dirichlet_model: nn.Module,
                      dataset_loader,
                      device='cpu',
                      threshold: Union[List[float], float] = None,
                      samples=1,
                      topk=None):
    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)
    exits_counter = defaultdict(int)

    for x, y in dataset_loader:
        model.eval()
        predictors.eval()
        dirichlet_model.eval()

        x, y = x.to(device), y.to(device)
        final_preds, preds = model(x)

        dirichlet = dirichlet_model(x)
        # dirichlet = nn.functional.softplus(dirichlet)
        posterior = Dirichlet(dirichlet)
        w = posterior.rsample()

        h = -w.log() * w
        # print(bi, i, h.sum())
        h = h / np.log(h.shape[-1])
        h = h.sum(-1)

        argmax = w.argmax(-1)

        logits = []
        for i, predictor in enumerate(predictors):
            if isinstance(predictor, BayesianHead):
                p = predictor(logits=preds[i],
                              branch_index=i, samples=samples)
            else:
                p = predictor(preds[i])
            logits.append(p)
        # logits.append(final_preds)

        for bi in range(x.shape[0]):
            _h = h[bi]
            if _h < 0.5:
                i = argmax[bi].item()
                p = logits[i][bi]
            else:
                i = 'final'
                p = final_preds[bi]
            # i = argmax[bi].item()
            # p = logits[i][bi]
            # for i, predictor in enumerate(predictors):
            #     p = logits[argmax[bi]][bi]  # .unsqueeze(0)
            #
            top_classes = torch.topk(p, p.size(-1))[1]
            pred_labels[i].append(top_classes.tolist())
            true_labels[i].append(y[bi].item())
            exits_counter[i] += 1

    #     true_labels.extend(y.tolist())
    #
    #     top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
    #     pred_labels['final'].extend(top_classes.tolist())
    #
    #     for i in range(len(preds)):
    #         p = predictors[i](preds[i])
    #         # pred = torch.argmax(pred, -1)
    #         top_classes = torch.topk(p, p.size(-1))[1]
    #         pred_labels[i].extend(top_classes.tolist())

    scores = {i: accuracy_score(true_labels[i], p, topk=topk)
              for i, p in pred_labels.items()}

    print(scores)
    print(exits_counter)
