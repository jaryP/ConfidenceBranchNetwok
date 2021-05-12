import logging
from collections import Sequence
from itertools import chain
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, kl_divergence, Dirichlet
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

from base.evaluators import standard_eval, branches_binary, branches_eval
from base.utils import mmd
from bayesian.bayesian_utils import compute_mmd, pdist
from bayesian.posteriors import BayesianPosterior, BayesianHead, BayesianHeads, \
    LayerEmbeddingBeta
from models.base import BranchModel, branches_predictions


def standard_trainer(model: BranchModel,
                     optimizer,
                     train_loader,
                     epochs,
                     scheduler=None,
                     early_stopping=None,
                     test_loader=None, eval_loader=None, device='cpu'):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0
    model.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            loss = nn.functional.cross_entropy(pred, y, reduction='none')
            losses.extend(loss.tolist())
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        # score_dict = {'Train score': train_scores, 'Test score': test_scores,
        #               'Eval score': eval_scores if eval_scores != 0 else 0}

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})

        scores.append((train_scores, eval_scores, test_scores))

    return best_model, \
           scores, \
           scores[best_model_i] if len(scores) > 0 else 0, \
           mean_losses


def joint_trainer(model: BranchModel,
                  predictors: nn.ModuleList,
                  optimizer,
                  train_loader,
                  epochs,
                  scheduler=None,
                  weights=None,
                  early_stopping=None,
                  test_loader=None, eval_loader=None, device='cpu'):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            predictors.train()

            x, y = x.to(device), y.to(device)
            final_pred, preds = model(x)
            loss = nn.functional.cross_entropy(final_pred, y,
                                               reduction='mean')
            for i, p in enumerate(preds):
                l = nn.functional.cross_entropy(predictors[i](p), y,
                                                reduction='mean')
                if weights is not None:
                    l *= weights[i]
                loss += l

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_predictors = predictors.state_dict()
                best_model_i = epoch
        else:
            best_predictors = predictors.state_dict()
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors), scores, scores[best_model_i] if len(
        scores) > 0 else 0, mean_losses


def output_combiner_trainer(model: BranchModel,
                            predictors: nn.ModuleList,
                            optimizer,
                            train_loader,
                            epochs,
                            convex_combination=False,
                            scheduler=None,
                            weights=None,
                            early_stopping=None,
                            test_loader=None, eval_loader=None, device='cpu'):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    if weights is not None:
        if not isinstance(weights, (torch.Tensor, torch.nn.Parameter)):
            weights = torch.tensor(weights)

        weights = weights.view(weights.shape[0], 1, 1).to(device)

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            predictors.train()

            x, y = x.to(device), y.to(device)
            final_pred, bos = model(x)

            preds = [p(bo) for p, bo in zip(predictors, bos)]
            # preds = torch.stack(preds, 0)

            preds = torch.stack(preds + [final_pred], 0)

            if weights is not None:
                w = weights
                if convex_combination:
                    w = torch.softmax(w, 0)
                preds = preds * w

            pred = preds.sum(0)

            loss = nn.functional.cross_entropy(pred, y, reduction='mean')

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores, _ = standard_eval(model, eval_loader, topk=[1, 5],
                                           device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_predictors = predictors.state_dict()
                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_predictors = predictors.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors), scores, scores[best_model_i] if len(
        scores) > 0 else 0, mean_losses


def binary_classifier_trainer(model: BranchModel,
                              predictors: nn.ModuleList,
                              binary_classifiers: nn.ModuleList,
                              optimizer,
                              train_loader,
                              epochs,
                              energy_w=1,
                              scheduler=None,
                              early_stopping=None,
                              test_loader=None, eval_loader=None, device='cpu'):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_binaries = binary_classifiers.state_dict()

    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    sample_image = next(iter(train_loader))[0][1:2].to(device)
    costs = branches_predictions(model, predictors, sample_image)

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            predictors.train()
            binary_classifiers.train()

            x, y = x.to(device), y.to(device)
            final_pred, bos = model(x)

            hs = torch.stack([h(bo)
                              for h, bo in zip(binary_classifiers, bos)], 0)

            # if hs.min() < 0 or hs.max() > 1:
            #     hs = torch.sigmoid(hs)

            preds = [p(bo) for p, bo in zip(predictors, bos)]
            preds = torch.stack(preds, 0)

            f_hat = final_pred
            gamma_hat = costs['final']

            for i in reversed(range(0, len(preds))):
                f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat
                gamma_hat = hs[i] * costs[i] + (1 - hs[i]) * gamma_hat

            # print(gamma_hat.mean())
            # print(gamma_hat.mean())
            # for y_b, y_c in zip(preds[-2::-1], hs[-2::-1]):
            #     f_hat = y_c * y_b + (1 - y_c[i]) * f_hat
            # gamma_hat = costs['final']
            # for i in reversed(range(0, len(preds))):
            # for y_b, y_c in zip(preds[-2::-1], hs[-2::-1]):
            #     f_hat = y_c * y_b + (1 - y_c[i]) * f_hat
            # print(gamma_hat.mean())
            gamma_hat = gamma_hat.mean() * energy_w

            loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')
            # print(gamma_hat, loss)
            loss += gamma_hat

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # s = branches_eval(model, predictors, test_loader, device=device)
        # print('Branches scores {}'.format(s))

        # s, c = branches_binary(model=model,
        #                        binary_classifiers=binary_classifiers,
        #                        dataset_loader=test_loader,
        #                        predictors=predictors,
        #                        threshold=0.5,
        #                        device=device)
        #
        # print(s, c)

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            # r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
            #     else early_stopping.step(mean_loss)
            r = early_stopping.step(mean_loss)
            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_predictors = predictors.state_dict()
                best_binaries = binary_classifiers.state_dict()

                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_predictors = predictors.state_dict()
            best_binaries = binary_classifiers.state_dict()

            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        branches_scores = branches_eval(model, predictors, test_loader,
                                        device=device)
        branches_scores = {k: v[1] for k, v in branches_scores.items()}

        s, c = branches_binary(model=model,
                               binary_classifiers=binary_classifiers,
                               dataset_loader=test_loader,
                               predictors=predictors,
                               threshold=0.5,
                               device=device)

        s = {k: v[1] for k, v in s.items()}
        print(s, c)
        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss, 'Branch test scores': branches_scores,
             's': s, 'c': c})

        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors, best_binaries), \
           scores, scores[best_model_i], mean_losses


def posterior_classifier_trainer(model: BranchModel,
                                 predictors: nn.ModuleList,
                                 posteriors: BayesianPosterior,
                                 optimizer,
                                 train_loader,
                                 epochs,
                                 prior: Union[Distribution, List[Distribution]],
                                 prior_w=1,
                                 energy_w=1e-8,
                                 use_mmd=True,
                                 scheduler=None,
                                 early_stopping=None,
                                 test_loader=None, eval_loader=None,
                                 cumulative_prior=False,
                                 device='cpu'):
    if not isinstance(prior, list):
        if not cumulative_prior:
            prior = [prior] * len(predictors)
    else:
        cumulative_prior = False

    sample_image = next(iter(train_loader))[0][1:2].to(device)
    costs = branches_predictions(model, predictors, sample_image)
    print(costs)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            posteriors.train()
            predictors.train()

            x, y = x.to(device), y.to(device)
            final_pred, bos = model(x)

            hs = torch.stack([posteriors(branch_index=j, logits=bo)
                              for j, bo
                              in enumerate(bos)], 0)

            ps = [posteriors.get_posterior(branch_index=j,
                                           logits=bo)
                  for j, bo
                  in enumerate(bos)]

            preds = [p(bo) for p, bo in zip(predictors, bos)]
            preds = torch.stack(preds, 0)

            f_hat = final_pred
            gamma_hat = costs['final']

            for i in reversed(range(0, len(preds))):
                f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat
                gamma_hat = hs[i] * costs[i] + (1 - hs[i]) * gamma_hat

            loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')
            losses.append(loss.item())

            gamma_hat = gamma_hat.mean() * energy_w
            loss += gamma_hat

            kl = 0

            if cumulative_prior:
                s = torch.cat([p.rsample() for i, p in enumerate(ps)], 0)
                sp = prior.sample(s.size()).to(device)
                kl = mmd(s, sp)
            else:
                for pi, p in enumerate(ps):
                    if use_mmd:
                        s = p.rsample()
                        kl += mmd(s, prior[pi].sample(s.size()).to(s.device))
                    else:
                        kl += kl_divergence(p, prior[pi]).mean()

            klw = 2 ** (len(train_loader) - i - 1) \
                  / (2 ** len(train_loader) - 1)
            kl *= prior_w  # * klw

            loss += kl

            # pairwise_kl = 0
            # for p in range(len(ps) - 1):
            #     # pairwise_kl += kl_divergence(ps[p], ps[p+1])
            #     # pairwise_kl += compute_mmd(ps[p].rsample([1, 100]),
            #     #                    ps[p+1].sample([1, 100]))
            #     pairwise_kl += torch.pow(ps[p].rsample([100]) -
            #                              ps[p + 1].sample([100]), 2).sum([1, 2])
            #
            # pairwise_kl = pairwise_kl.mean()
            #
            # # print(pairwise_kl, end='... ')
            # pairwise_kl = 1 / pairwise_kl.mean()
            # # pairwise_kl *= 1e-6
            # # print(pairwise_kl)
            #
            # loss += pairwise_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for optimizer.
            # for n, p in posteriors.named_parameters():
            #     print(n, p.grad)

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores, _ = standard_eval(model, eval_loader, topk=[1, 5],
                                           device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return best_model, scores, scores[best_model_i], mean_losses


# def posterior_regularization_trainer(model: BranchModel,
#                                      predictors: nn.ModuleList,
#                                      posteriors: BayesianPosterior,
#                                      optimizer,
#                                      train_loader,
#                                      epochs,
#                                      prior: Union[
#                                          Distribution, List[Distribution]],
#                                      kl_divergence_w=1,
#                                      scheduler=None,
#                                      early_stopping=None,
#                                      test_loader=None, eval_loader=None,
#                                      device='cpu'):
#     if not isinstance(prior, list):
#         prior = [prior] * len(predictors)
#
#     scores = []
#     mean_losses = []
#
#     best_model = model.state_dict()
#     best_model_i = 0
#
#     model.to(device)
#     predictors.to(device)
#
#     if early_stopping is not None:
#         early_stopping.reset()
#
#     model.train()
#     bar = tqdm(range(epochs), leave=True)
#     for epoch in bar:
#         model.train()
#         losses = []
#         for i, (x, y) in enumerate(train_loader):
#             model.train()
#             posteriors.train()
#             predictors.train()
#
#             x, y = x.to(device), y.to(device)
#             final_pred, bos = model(x)
#
#             hs = torch.stack([posteriors(branch_index=i, logits=bo)
#                               for i, bo
#                               in enumerate(bos)], 0)
#
#             # hs = hs.to(device)
#             # if hs.min() < 0 or hs.max() > 1:
#             #     hs = torch.sigmoid(hs)
#
#             preds = [p(bo) for p, bo in zip(predictors, bos)]
#             preds = torch.stack(preds + [final_pred], 0)
#
#             f_hat = preds[-1]
#             for i in reversed(range(0, len(preds) - 1)):
#                 f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat
#
#             loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')
#
#             ps = [posteriors.get_posterior(branch_index=i,
#                                            logits=bo)
#                   for i, bo
#                   in enumerate(bos)]
#
#             kl = 0
#             for i, p in enumerate(ps):
#                 print('BETA', i,
#                       p.concentration1.shape,
#                       p.concentration1[0],
#                       p.concentration0[0],
#                       hs[i][0],
#                       kl_divergence(p, prior[i]).mean())
#
#                 kl += kl_divergence(p, prior[i]).mean()
#             kl *= kl_divergence_w
#
#             print(kl)
#
#             loss += kl
#
#             losses.append(loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # for optimizer.
#             # for n, p in posteriors.named_parameters():
#             #     print(n, p.grad)
#
#         if scheduler is not None:
#             if isinstance(scheduler, (StepLR, MultiStepLR)):
#                 scheduler.step()
#             elif hasattr(scheduler, 'step'):
#                 scheduler.step()
#
#         if eval_loader is not None:
#             eval_scores, _ = standard_eval(model, eval_loader, topk=[1, 5],
#                                            device=device)
#         else:
#             eval_scores = 0
#
#         mean_loss = sum(losses) / len(losses)
#         mean_losses.append(mean_loss)
#
#         if early_stopping is not None:
#             r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
#                 else early_stopping.step(mean_loss)
#
#             if r < 0:
#                 break
#             elif r > 0:
#                 best_model = model.state_dict()
#                 best_model_i = epoch
#         else:
#             best_model = model.state_dict()
#             best_model_i = epoch
#
#         train_scores = standard_eval(model, train_loader, device=device)
#         test_scores = standard_eval(model, test_loader, device=device)
#
#         bar.set_postfix(
#             {'Train score': train_scores, 'Test score': test_scores,
#              'Eval score': eval_scores if eval_scores != 0 else 0,
#              'Mean loss': mean_loss})
#         scores.append((train_scores, eval_scores, test_scores))
#
#     return best_model, scores, scores[best_model_i], mean_losses


def binary_posterior_joint_trainer(model: BranchModel,
                                   predictors: nn.ModuleList,
                                   posteriors: BayesianPosterior,
                                   optimizer,
                                   train_loader,
                                   epochs,
                                   prior: Union[
                                       Distribution, List[Distribution]],
                                   prior_w=1,
                                   energy_w=1e-3,
                                   use_mmd=True,
                                   scheduler=None,
                                   early_stopping=None,
                                   test_loader=None, eval_loader=None,
                                   cumulative_prior=False,
                                   device='cpu'):
    if not isinstance(prior, list):
        if not cumulative_prior:
            prior = [prior] * len(predictors)
    else:
        cumulative_prior = False

    sample_image = next(iter(train_loader))[0][1:2].to(device)
    costs = branches_predictions(model, predictors, sample_image)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            posteriors.train()
            predictors.train()

            x, y = x.to(device), y.to(device)
            final_pred, bos = model(x)

            hs = torch.stack([posteriors(branch_index=j, logits=bo)
                              for j, bo
                              in enumerate(bos)], 0)

            ps = [posteriors.get_posterior(branch_index=j,
                                           logits=bo)
                  for j, bo
                  in enumerate(bos)]

            # hs = hs.to(device)
            # if hs.min() < 0 or hs.max() > 1:
            #     hs = torch.sigmoid(hs)

            preds = torch.stack([p(bo) * h for p, bo, h in
                                 zip(predictors, bos, hs)])

            preds = preds.sum(0) + final_pred
            # preds = torch.stack(preds + [final_pred], 0)
            #
            f_hat = preds[-1]
            gamma_hat = costs['final']

            for pi in reversed(range(0, len(preds) - 1)):
                f_hat = hs[pi] * preds[pi] + (1 - hs[pi]) * f_hat
                gamma_hat = hs[i] * costs[i] + (1 - hs[i]) * gamma_hat

            gamma_loss = gamma_hat * energy_w
            loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')

            kl = 0

            if cumulative_prior:
                s = torch.cat([p.rsample() for i, p in enumerate(ps)], 0)
                sp = prior.sample(s.size()).to(device)
                kl = mmd(s, sp)
            else:
                for pi, p in enumerate(ps):
                    # a, b = p.concentration1.mean(0), p.concentration0.mean(0)
                    # m = a / (a + b)
                    # std = (a * b) / ((a + b)**2 * (a + b + 1))
                    # print('BETA', i,
                    #       p.concentration1.mean(0).item(),
                    #       p.concentration0.mean(0).item(),
                    #       hs[i].mean(0).item(), hs[i].std(0).item(),
                    #       kl_divergence(p, prior[i]).mean().item(),
                    #       m, std)
                    # print(mmd(s, prior[i].sample(s.size()).to(s.device)))
                    if use_mmd:
                        s = p.rsample()
                        kl += mmd(s, prior[pi].sample(s.size()).to(s.device))
                    else:
                        kl += kl_divergence(p, prior[pi]).mean()

            # a = 0
            # for pi in range(len(ps) - 1):
            #     # a += kl_divergence(ps[pi], ps[pi + 1]).sum()
            #     a += torch.pow(hs[pi] - hs[pi + 1], 2).mean()
            # # print(a, 1 / a)
            # a = 1 / a

            klw = 2 ** (len(train_loader) - i - 1) \
                  / (2 ** len(train_loader) - 1)

            kl *= prior_w * klw

            losses.append(loss.item())

            loss += kl + gamma_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for optimizer.
            # for n, p in posteriors.named_parameters():
            #     print(n, p.grad)

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return best_model, scores, scores[best_model_i], mean_losses


# def posterior_classifier_trainer(model: BranchModel,
#                                  predictors: nn.ModuleList,
#                                  posteriors: BayesianPosterior,
#                                  optimizer,
#                                  train_loader,
#                                  epochs,
#                                  prior: Union[Distribution, List[Distribution]],
#                                  prior_w=1,
#                                  entropy_w=1,
#                                  use_mmd=True,
#                                  scheduler=None,
#                                  early_stopping=None,
#                                  test_loader=None, eval_loader=None,
#                                  cumulative_prior=False,
#                                  device='cpu'):
#     if not isinstance(prior, list):
#         if not cumulative_prior:
#             prior = [prior] * len(predictors)
#     else:
#         cumulative_prior = False
#
#     scores = []
#     mean_losses = []
#
#     best_model = model.state_dict()
#     best_predictors = predictors.state_dict()
#     best_posteriors = posteriors.state_dict()
#     best_model_i = 0
#
#     model.to(device)
#     predictors.to(device)
#
#     if early_stopping is not None:
#         early_stopping.reset()
#
#     model.train()
#     bar = tqdm(range(epochs), leave=True)
#     for epoch in bar:
#         model.train()
#         losses = []
#
#         for i, (x, y) in enumerate(train_loader):
#             model.train()
#             posteriors.train()
#             predictors.train()
#
#             x, y = x.to(device), y.to(device)
#             final_pred, bos = model(x)
#
#             hs = torch.stack([posteriors(branch_index=j, logits=bo)
#                               for j, bo
#                               in enumerate(bos)], 0)
#             # hs = hs.to(device)
#             # if hs.min() < 0 or hs.max() > 1:
#             #     hs = torch.sigmoid(hs)
#
#             preds = [p(bo) for p, bo in zip(predictors, bos)]
#             preds = torch.stack(preds + [final_pred], 0)
#
#             f_hat = preds[-1]
#             for pi in reversed(range(0, len(preds) - 1)):
#                 f_hat = hs[pi] * preds[pi] + (1 - hs[pi]) * f_hat
#
#             loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')
#
#             ps = [posteriors.get_posterior(branch_index=j,
#                                            logits=bo)
#                   for j, bo
#                   in enumerate(bos)]
#
#             kl = 0
#
#             if cumulative_prior:
#                 s = torch.cat([p.rsample() for i, p in enumerate(ps)], 0)
#                 sp = prior.sample(s.size()).to(device)
#                 kl = mmd(s, sp)
#             else:
#                 for pi, p in enumerate(ps):
#                     # a, b = p.concentration1.mean(0), p.concentration0.mean(0)
#                     # m = a / (a + b)
#                     # std = (a * b) / ((a + b)**2 * (a + b + 1))
#                     # print('BETA', i,
#                     #       p.concentration1.mean(0).item(),
#                     #       p.concentration0.mean(0).item(),
#                     #       hs[i].mean(0).item(), hs[i].std(0).item(),
#                     #       kl_divergence(p, prior[i]).mean().item(),
#                     #       m, std)
#                     # print(mmd(s, prior[i].sample(s.size()).to(s.device)))
#                     if use_mmd:
#                         s = p.rsample()
#                         # kl += mmd(s, prior[pi].sample(s.size()).to(s.device))
#                         kl += compute_mmd(s, prior[pi].sample(s.size()).to(
#                             s.device))
#                     else:
#                         kl += kl_divergence(p, prior[pi]).mean()
#
#             klw = 2 ** (len(train_loader) - i - 1) \
#                   / (2 ** len(train_loader) - 1)
#
#             kl *= prior_w
#
#             h = torch.stack([p.entropy().mean(0) for p in ps])
#
#             if i == 0:
#                 print(h)
#
#             h = torch.sum(h) * entropy_w
#
#             reg = (h + kl) * klw
#             loss += reg
#
#             losses.append(loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # for optimizer.
#             # for n, p in posteriors.named_parameters():
#             #     print(n, p.grad)
#
#         ps = [posteriors.get_posterior(branch_index=j,
#                                        logits=bo)
#               for j, bo
#               in enumerate(bos)]
#         h = [p.entropy().mean(0).item() for p in ps]
#         print(epoch, h)
#
#         if scheduler is not None:
#             if isinstance(scheduler, (StepLR, MultiStepLR)):
#                 scheduler.step()
#             elif hasattr(scheduler, 'step'):
#                 scheduler.step()
#
#         if eval_loader is not None:
#             eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
#                                         device=device)
#         else:
#             eval_scores = 0
#
#         mean_loss = sum(losses) / len(losses)
#         mean_losses.append(mean_loss)
#
#         if early_stopping is not None:
#             r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
#                 else early_stopping.step(mean_loss)
#             if r < 0:
#                 break
#             elif r > 0:
#                 best_model = model.state_dict()
#                 best_predictors = predictors.state_dict()
#                 best_posteriors = posteriors.state_dict()
#                 best_model_i = epoch
#         else:
#             best_model = model.state_dict()
#             best_predictors = predictors.state_dict()
#             best_posteriors = posteriors.state_dict()
#             best_model_i = epoch
#
#         train_scores = standard_eval(model, train_loader, device=device)
#         test_scores = standard_eval(model, test_loader, device=device)
#
#         bar.set_postfix(
#             {'Train score': train_scores, 'Test score': test_scores,
#              'Eval score': eval_scores if eval_scores != 0 else 0,
#              'Mean loss': mean_loss})
#         scores.append((train_scores, eval_scores, test_scores))
#
#     return (best_model, best_predictors, best_posteriors), \
#            scores, scores[best_model_i], mean_losses


def bayesian_joint_trainer(model: BranchModel,
                           predictors: BayesianHeads,
                           optimizer,
                           train_loader,
                           epochs,
                           scheduler=None,
                           weights=None,
                           early_stopping=None,
                           samples=1,
                           prior_w=1,
                           test_loader=None,
                           eval_loader=None,
                           device='cpu'):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:

        losses = []
        for i, (x, y) in enumerate(train_loader):
            batch_w = 2 ** (len(train_loader) - i - 1) \
                      / (2 ** len(train_loader) - 1)

            model.train()
            predictors.train()

            x, y = x.to(device), y.to(device)
            final_pred, preds = model(x)
            # loss = nn.functional.cross_entropy(final_pred, y,
            #                                    reduction='mean')
            loss = 0
            for i, p in enumerate(preds):
                o, prior_loss = predictors(logits=p, branch_index=i,
                                           samples=samples)
                l = nn.functional.cross_entropy(o, y,
                                                reduction='mean')
                l = l + prior_loss * prior_w * batch_w
                # print(l)

                if weights is not None:
                    l *= weights[i]

                loss += l

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            """ r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)"""
            r = early_stopping.step(mean_loss)
            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_predictors = predictors.state_dict()
                best_model_i = epoch
        else:
            best_predictors = predictors.state_dict()
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        branches_scores = branches_eval(model, predictors, test_loader,
                                        device=device)
        print(branches_scores)
        branches_scores = {k: v[1] for k, v in branches_scores.items()}

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss, 'Branch scores': branches_scores})
        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors), scores, scores[best_model_i] if len(
        scores) > 0 else 0, mean_losses


# def posterior_regularization_trainer(model: BranchModel,
#                                      predictors: nn.ModuleList,
#                                      posteriors: BayesianPosterior,
#                                      optimizer,
#                                      train_loader,
#                                      epochs,
#                                      prior: Union[
#                                          Distribution, List[Distribution]],
#                                      kl_divergence_w=1,
#                                      scheduler=None,
#                                      early_stopping=None,
#                                      test_loader=None, eval_loader=None,
#                                      device='cpu'):
#     if not isinstance(prior, list):
#         prior = [prior] * len(predictors)
#
#     scores = []
#     mean_losses = []
#
#     best_model = model.state_dict()
#     best_model_i = 0
#
#     model.to(device)
#     predictors.to(device)
#
#     if early_stopping is not None:
#         early_stopping.reset()
#
#     model.train()
#     bar = tqdm(range(epochs), leave=True)
#     for epoch in bar:
#         model.train()
#         losses = []
#         for i, (x, y) in enumerate(train_loader):
#             model.train()
#             posteriors.train()
#             predictors.train()
#
#             x, y = x.to(device), y.to(device)
#             final_pred, bos = model(x)
#
#             hs = torch.stack([posteriors(branch_index=i, logits=bo)
#                               for i, bo
#                               in enumerate(bos)], 0)
#
#             # hs = hs.to(device)
#             # if hs.min() < 0 or hs.max() > 1:
#             #     hs = torch.sigmoid(hs)
#
#             preds = [p(bo) for p, bo in zip(predictors, bos)]
#             preds = torch.stack(preds + [final_pred], 0)
#
#             f_hat = preds[-1]
#             for i in reversed(range(0, len(preds) - 1)):
#                 f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat
#
#             loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')
#
#             ps = [posteriors.get_posterior(branch_index=i,
#                                            logits=bo)
#                   for i, bo
#                   in enumerate(bos)]
#
#             kl = 0
#             for i, p in enumerate(ps):
#                 print('BETA', i,
#                       p.concentration1.shape,
#                       p.concentration1[0],
#                       p.concentration0[0],
#                       hs[i][0],
#                       kl_divergence(p, prior[i]).mean())
#
#                 kl += kl_divergence(p, prior[i]).mean()
#             kl *= kl_divergence_w
#
#             print(kl)
#
#             loss += kl
#
#             losses.append(loss.item())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # for optimizer.
#             # for n, p in posteriors.named_parameters():
#             #     print(n, p.grad)
#
#         if scheduler is not None:
#             if isinstance(scheduler, (StepLR, MultiStepLR)):
#                 scheduler.step()
#             elif hasattr(scheduler, 'step'):
#                 scheduler.step()
#
#         if eval_loader is not None:
#             eval_scores, _ = standard_eval(model, eval_loader, topk=[1, 5],
#                                            device=device)
#         else:
#             eval_scores = 0
#
#         mean_loss = sum(losses) / len(losses)
#         mean_losses.append(mean_loss)
#
#         if early_stopping is not None:
#             r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
#                 else early_stopping.step(mean_loss)
#
#             if r < 0:
#                 break
#             elif r > 0:
#                 best_model = model.state_dict()
#                 best_model_i = epoch
#         else:
#             best_model = model.state_dict()
#             best_model_i = epoch
#
#         train_scores = standard_eval(model, train_loader, device=device)
#         test_scores = standard_eval(model, test_loader, device=device)
#
#         bar.set_postfix(
#             {'Train score': train_scores, 'Test score': test_scores,
#              'Eval score': eval_scores if eval_scores != 0 else 0,
#              'Mean loss': mean_loss})
#         scores.append((train_scores, eval_scores, test_scores))
#
#     return best_model, scores, scores[best_model_i], mean_losses


def branching_trainer(model: BranchModel,
                      predictors: nn.ModuleList,
                      binary_classifiers: LayerEmbeddingBeta,
                      optimizer,
                      train_loader,
                      epochs,
                      one_hot=False,
                      energy_reg=1,
                      samples=1,
                      scheduler=None,
                      early_stopping=None,
                      test_loader=None, eval_loader=None, device='cpu'):
    logger = logging.getLogger(__name__)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_binaries = binary_classifiers.state_dict()

    best_model_i = 0

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    sample_image = next(iter(train_loader))[0][1:2].to(device)
    costs_d = branches_predictions(model, predictors, sample_image)

    costs = torch.empty(len(costs_d), device=device)
    for k, v in costs_d.items():
        if k == 'final':
            k = len(costs_d) - 1
        costs[k] = v
    # print(costs)
    # print(1/costs, nn.functional.softmax((-costs)/2))

    costs = nn.functional.softmax(-costs, 0)
    p = [10, 8, 5.0, 3, 1.5, 1.5]
    # p = costs.tolist()
    prior = Dirichlet(torch.tensor(p, device=device))

    logger.info('Prior parameters: {}'.format(p))
    logger.info('Prior means: {}'.format(prior.mean.tolist()))
    logger.info('Prior variances: {}'.format(prior.variance.tolist()))

    logger.info('Prior costs: {}'.format(costs))
    costs = costs.unsqueeze(-1).unsqueeze(-1)

    bar = tqdm(range(epochs), leave=True)

    for epoch in bar:

        losses = []
        kl_losses = []

        for i, (x, y) in enumerate(train_loader):

            model.train()
            predictors.train()
            binary_classifiers.train()

            x, y = x.to(device), y.to(device)
            final_pred, bos = model(x)

            hs = torch.stack([binary_classifiers(branch_index=j, logits=bo,
                                                 samples=samples)
                              for j, bo
                              in enumerate(bos)], 0)
            hs = torch.cat(
                (hs, torch.ones(1, final_pred.size(0), 1, device=device)), 0)

            # print(hs[:, 10].tolist())
            #
            # ps = [binary_classifiers.get_posterior(branch_index=j,
            #                                        logits=bo).concentration0[10].item()
            #       for j, bo
            #       in enumerate(bos)]
            #
            # print(ps)

            # mask = torch.zeros_like(hs)
            # mask[0] = 1

            # posteriors = torch.stack([binary_classifiers.get_posterior(branch_index=j, logits=bo)
            #                   for j, bo
            #                   in enumerate(bos)], 0)
            # print(hs.mean(1))

            # ones = torch.ones((1, hs.size(1), 1), device=device)
            # hs = torch.cat((hs, ones), 0)

            # hs[0, :] = 1
            # cp = torch.cumprod(1 - hs, 0)[1:, :]
            # print(cp[1:, 10])

            a, b = torch.split(hs, [len(hs) - 1, 1], 0)

            cp = torch.cumprod(1 - a, 0)

            ones = torch.ones((1, cp.size(1), 1), device=device)
            cp = torch.cat((ones, cp), 0)
            hs = hs * cp

            # print(kl)
            # kl = torch.nn.functional.kl_div(hs.transpose(0, 1).squeeze(-1),
            #                                 prior_sample,
            #                                 reduction='batchmean')

            # print(cp[:, 10])
            # print(hs[:, 10].tolist())
            # print(hs.mean(1).tolist(), hs.std(1).tolist())
            # print()

            # print(hs[:, 10], hs[:, 10].sum())
            # input()

            # _hs = torch.zeros_like(hs) * mask + hs * (1 - mask)
            # print(_hs[:, 10])

            # cp = torch.cumprod(1 - _hs, 0)
            # print(cp[:, 10])
            # cp = torch.roll(cp, 0, 1)
            # cp = cp * hs
            # # cp1 = torch.roll(torch.cumprod(1 - hs, 0), 0, 1)
            #
            # #  We use a mask because the first value should not be changed
            #
            # hs = hs * mask + cp * (1 - mask)
            # print(hs[:, 10], hs[:, 10].sum())
            # input()

            # print(cp.sum(0)[0])
            # a = cp1[:, 0]
            # b = hs[:, 0]
            # b = a.sum()
            # prova = 0

            # cp = hs
            # cp[1:] = hs[1:] * torch.cumprod(1 - hs[:-1], -1)
            # print(torch.cumprod(1 - hs[:-1], -1)[0, 0])

            # if hs.min() < 0 or hs.max() > 1:
            #     hs = torch.sigmoid(hs)

            # print(hs[:, 0:2].tolist())

            # preds = [p(bo) for p, bo in zip(predictors, bos)] + [final_pred]
            # preds = torch.stack(preds, 0)
            # preds = preds * hs
            # f_hat = preds.sum(0)  # + final_pred

            def gaussian_kernel(a, b):
                dim1_1, dim1_2 = a.shape[0], b.shape[0]
                depth = a.shape[1]
                a = a.view(dim1_1, 1, depth)
                b = b.view(1, dim1_2, depth)
                a_core = a.expand(dim1_1, dim1_2, depth)
                b_core = b.expand(dim1_1, dim1_2, depth)
                numerator = (a_core - b_core).pow(2).mean(2) / depth
                return torch.exp(-numerator)

            def mmd(x, y):
                x_kernel = gaussian_kernel(x, x)
                y_kernel = gaussian_kernel(y, y)
                xy_kernel = gaussian_kernel(x, y)
                return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

            costs_loss = 0
            if energy_reg > 0:
                prior_sample = prior.sample([hs.size(1)])
                _hs = hs
                a = mmd(_hs.transpose(0, 1).squeeze(-1), prior_sample)
                # print(a)
                #
                # # print(prior_sample[0], hs.transpose(0, 1)[0])
                # _hs = hs.contiguous().transpose(0, 1)
                # a = compute_mmd(_hs, prior_sample.unsqueeze(-1))
                # print(a.shape)
                kl = a
                # print(a.mean())

                # a = pdist(hs.transpose(0, 1).squeeze(-1).unsqueeze(1), prior_sample.unsqueeze(1))
                # input(a.shape)
                # prior_sample = costs
                #
                # _hs = hs.transpose(0, 1).squeeze(-1)
                # div = _hs / prior_sample
                # div = div.log()
                # kl = _hs * div
                # kl = kl.sum(1)
                # kl = kl.mean()
                # print(kl)

                costs_loss = kl * energy_reg
                kl_losses.append(costs_loss.item())
            else:
                kl_losses.append(costs_loss)
            # print(costs_loss)

            # f_hat = final_pred
            #
            # gamma_hat = costs['final']
            # for i in reversed(range(0, len(preds))):
            #     f_hat = cp[i] * preds[i] + (1 - cp[i]) * f_hat
            #     # gamma_hat = hs[i] * costs[i] + (1 - hs[i]) * gamma_hat
            #
            # print(gamma_hat.mean())
            # print(gamma_hat.mean())
            # for y_b, y_c in zip(preds[-2::-1], hs[-2::-1]):
            #     f_hat = y_c * y_b + (1 - y_c[i]) * f_hat
            # gamma_hat = costs['final']
            # for i in reversed(range(0, len(preds))):
            # for y_b, y_c in zip(preds[-2::-1], hs[-2::-1]):
            #     f_hat = y_c * y_b + (1 - y_c[i]) * f_hat
            # print(gamma_hat.mean())
            # gamma_hat = gamma_hat.mean() * energy_w

            if one_hot:
                index = hs.max(dim=0, keepdims=True)[1]
                hs_hard = torch.zeros_like(hs).scatter_(0, index, 1.0)
                hs = hs_hard - hs.detach() + hs
                # print(hs_hard.shape, hs_hard[:, 0:2].tolist())

            alls = [p(bo) for p, bo in zip(predictors, bos)] + [final_pred]
            loss = torch.stack([nn.functional.cross_entropy(p, y,
                                                            reduction='none')
                                for p in alls], 0)
            loss = loss * hs.squeeze(-1)
            loss = loss.mean(-1).sum(0)
            # # # print(loss.shape, hs.shape)
            # #
            # loss = loss.mean(1).mean()
            # print(loss)

            loss += costs_loss

            # print(gamma_hat, loss)
            # loss += gamma_hat

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for n, m in chain(model.named_parameters(),
            #       predictors.named_parameters(),
            #       binary_classifiers.named_parameters()):
            #     print(n, getattr(m, 'grad', None) is None)
            # input()

        # s = branches_eval(model, predictors, test_loader, device=device)
        # print('Branches scores {}'.format(s))

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)
        mean_kl_loss = sum(kl_losses) / len(kl_losses)

        if early_stopping is not None:
            # r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
            #     else early_stopping.step(mean_loss)
            r = early_stopping.step(mean_loss)
            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_predictors = predictors.state_dict()
                best_binaries = binary_classifiers.state_dict()

                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_predictors = predictors.state_dict()
            best_binaries = binary_classifiers.state_dict()

            best_model_i = epoch

        train_scores = standard_eval(model, train_loader, device=device)
        test_scores = standard_eval(model, test_loader, device=device)

        branches_scores = branches_eval(model, predictors, test_loader,
                                        device=device)
        branches_scores = {k: v[1] for k, v in branches_scores.items()}

        s, c = branches_binary(model=model,
                               binary_classifiers=binary_classifiers,
                               dataset_loader=test_loader,
                               predictors=predictors,
                               threshold=0.5,
                               device=device)

        s = {k: v.get(1, 0) for k, v in s.items()}
        print(s, c)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss, 'KL loss': mean_kl_loss,
             'Branch test scores': branches_scores,
             's': s, 'c': c})

        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors, best_binaries), \
           scores, scores[best_model_i], mean_losses
