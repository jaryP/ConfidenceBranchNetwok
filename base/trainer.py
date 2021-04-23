from collections import Sequence
from typing import List, Union

import torch
from torch import nn
from torch.distributions import Distribution, kl_divergence
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

from base.evaluators import standard_eval
from base.utils import mmd
from bayesian.posteriors import BayesianPosterior
from models.base import BranchModel


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
                              scheduler=None,
                              early_stopping=None,
                              test_loader=None, eval_loader=None, device='cpu'):
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
            for i in reversed(range(0, len(preds) - 1)):
                f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat

            loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')

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


def posterior_classifier_trainer(model: BranchModel,
                                 predictors: nn.ModuleList,
                                 posteriors: BayesianPosterior,
                                 optimizer,
                                 train_loader,
                                 epochs,
                                 prior: Union[Distribution, List[Distribution]],
                                 prior_w=1,
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

            preds = [p(bo) for p, bo in zip(predictors, bos)]
            preds = torch.stack(preds + [final_pred], 0)

            f_hat = preds[-1]
            for pi in reversed(range(0, len(preds) - 1)):
                f_hat = hs[pi] * preds[pi] + (1 - hs[pi]) * f_hat

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

            # kl = kl / len(ps)
            klw = 2 ** (len(train_loader) - i - 1) \
                  / (2 ** len(train_loader) - 1)
            kl *= prior_w * klw

            # print(kl, loss, klw)
            losses.append(loss.item())

            loss += kl

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


def posterior_regularization_trainer(model: BranchModel,
                                     predictors: nn.ModuleList,
                                     posteriors: BayesianPosterior,
                                     optimizer,
                                     train_loader,
                                     epochs,
                                     prior: Union[
                                         Distribution, List[Distribution]],
                                     kl_divergence_w=1,
                                     scheduler=None,
                                     early_stopping=None,
                                     test_loader=None, eval_loader=None,
                                     device='cpu'):
    if not isinstance(prior, list):
        prior = [prior] * len(predictors)

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

            hs = torch.stack([posteriors(branch_index=i, logits=bo)
                              for i, bo
                              in enumerate(bos)], 0)

            # hs = hs.to(device)
            # if hs.min() < 0 or hs.max() > 1:
            #     hs = torch.sigmoid(hs)

            preds = [p(bo) for p, bo in zip(predictors, bos)]
            preds = torch.stack(preds + [final_pred], 0)

            f_hat = preds[-1]
            for i in reversed(range(0, len(preds) - 1)):
                f_hat = hs[i] * preds[i] + (1 - hs[i]) * f_hat

            loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')

            ps = [posteriors.get_posterior(branch_index=i,
                                           logits=bo)
                  for i, bo
                  in enumerate(bos)]

            kl = 0
            for i, p in enumerate(ps):
                print('BETA', i,
                      p.concentration1.shape,
                      p.concentration1[0],
                      p.concentration0[0],
                      hs[i][0],
                      kl_divergence(p, prior[i]).mean())

                kl += kl_divergence(p, prior[i]).mean()
            kl *= kl_divergence_w

            print(kl)

            loss += kl

            losses.append(loss.item())

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
