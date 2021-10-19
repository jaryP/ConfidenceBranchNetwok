import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence, Bernoulli, ContinuousBernoulli
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

from base.evaluators import standard_eval, branches_eval, binary_eval
from models.base import BranchModel
from utils import get_device


def standard_trainer(model: BranchModel,
                     predictors: nn.Module,
                     optimizer,
                     train_loader,
                     epochs,
                     scheduler=None,
                     early_stopping=None,
                     test_loader=None, eval_loader=None):
    device = get_device(model)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0
    best_eval_score = -1

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
            pred = model(x)[-1]
            pred = predictors[-1].logits(pred)

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
            eval_score = standard_eval(model, predictors[-1],
                                       eval_loader)
        else:
            eval_score = None

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_score) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = (model.state_dict(), predictors.state_dict())
                best_model_i = epoch
        else:
            if (eval_score is not None and eval_score > best_eval_score) \
                    or eval_score is None:
                if eval_score is not None:
                    best_eval_score = eval_score

                best_model = (model.state_dict(), predictors.state_dict())

                best_model_i = epoch

        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_score if eval_score != 0 else 0,
             'Mean loss': mean_loss})

        scores.append((train_scores, eval_score, test_scores))

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
                  joint_type='predictions',
                  early_stopping=None,
                  test_loader=None, eval_loader=None):
    device = get_device(model)

    if joint_type not in ['losses', 'predictions']:
        raise ValueError

    if weights is None:
        weights = torch.tensor([1.0] * model.n_branches(), device=device)

    if not isinstance(weights, (torch.Tensor, torch.nn.Parameter)):
        if isinstance(weights, (int, float)):
            weights = torch.tensor([weights] * model.n_branches(),
                                   device=device, dtype=torch.float)

        else:
            weights = torch.tensor(weights, device=device, dtype=torch.float)

        if joint_type == 'predictions':
            weights = weights.unsqueeze(-1)
            weights = weights.unsqueeze(-1)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_model_i = 0
    best_eval_score = -1

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
            # klw = 2 ** (len(train_loader) - i - 1) \
            #       / (2 ** len(train_loader) - 1)

            preds = model(x)
            logits = []

            for j, bo in enumerate(preds):
                l = predictors[j].logits(bo)
                logits.append(l)

            preds = torch.stack(logits, 0)

            if joint_type == 'predictions':
                preds = weights * preds
                f_hat = preds.sum(0)

                loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')

            else:
                loss = torch.stack(
                    [nn.functional.cross_entropy(p, y, reduction='mean')
                     for p in preds], 0)
                loss = loss * weights
                loss = loss.sum()

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
            """eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)"""
            branches_scores = branches_eval(model, predictors, eval_loader)
            # branches_scores = {k: v[1] for k, v in branches_scores.items()}
            eval_scores = branches_scores['final']
        else:
            eval_scores = None

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
            if (eval_scores is not None and eval_scores.get(1,
                                                            0) > best_eval_score) or eval_scores is None:
                if eval_scores is not None:
                    best_eval_score = eval_scores.get(1, 0)

                best_model = model.state_dict()
                best_predictors = predictors.state_dict()

                best_model_i = epoch

        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])

        scores.append(test_scores)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors), scores, scores[best_model_i] if len(
        scores) > 0 else 0, mean_losses


def binary_bernulli_trainer(model: BranchModel,
                            predictors: nn.ModuleList,
                            bernulli_models: nn.ModuleList,
                            optimizer,
                            train_loader,
                            epochs,
                            prior_parameters,
                            prior_w=1e-3,
                            fixed_bernulli=False,
                            joint_type='predictions',
                            sample=True,
                            scheduler=None,
                            early_stopping=None,
                            test_loader=None,
                            eval_loader=None,
                            recursive=False):

    device = get_device(model)

    if joint_type not in ['losses', 'predictions']:
        raise ValueError

    if isinstance(prior_parameters, (float, int)):
        prior_parameters = [prior_parameters] * (len(predictors) - 1)

    beta_priors = []

    for p in prior_parameters:
        beta = ContinuousBernoulli(p)
        beta_priors.append(beta)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0

    model.to(device)
    predictors.to(device)
    bernulli_models.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        predictors.train()
        bernulli_models.train()

        losses = []
        kl_losses = []

        for bi, (x, y) in tqdm(enumerate(train_loader), leave=False,
                               total=len(train_loader)):

            x, y = x.to(device), y.to(device)
            bos = model(x)

            distributions, logits = [], []

            for j, bo in enumerate(bos):
                l, b = predictors[j](bo)
                distributions.append(b)
                logits.append(l)

            preds = torch.stack(logits, 1)
            distributions = torch.stack(distributions, 1)

            kl = 0

            if not fixed_bernulli:

                # PROVA CON KL ADATTIVO

                with torch.no_grad():
                    _y = y.unsqueeze(1)
                    mx = torch.argmax(preds, -1)
                    a = (mx == _y).float()

                    if True:
                        sf = torch.softmax(preds, -1)
                        mx, _ = torch.max(sf, -1)
                        a = a * mx

                    # ones = torch.zeros_like(a)
                    # ones[:, -1] = 1
                    # a = torch.maximum(a, ones)

                    a = a.unsqueeze(-1)

                bce = nn.functional.binary_cross_entropy(
                    distributions, a, reduction='none')

                kl = bce.sum(-1)

                # a = torch.maximum(a, prior_stack)
                # d = ContinuousBernoulli(distributions)
                # p = ContinuousBernoulli(a)
                # kl = kl_divergence(d, p).sum(1)

                # d = ContinuousBernoulli(torch.stack(distributions, 1)
                #                         .squeeze(-1))
                # # (pred == y).sum().item()
                # _kl = kl_divergence(d, ContinuousBernoulli(a))
                #
                # kl = _kl.sum(-1)

                # for d, p in zip(distributions, beta_priors):
                #     d = ContinuousBernoulli(d)
                #
                #     _kl = kl_divergence(d, p).squeeze()
                #     kl += _kl

                kl = kl.mean() * prior_w
                # kl = kl.mean()

                kl_losses.append(kl.item())

                # klw = 2 ** (len(train_loader) - bi - 1) / \
                #       (2 ** len(train_loader) - 1)
                #
                # # klw = 2 ** (epochs - epoch - 1) \
                # #       / (2 ** epochs - 1)
                # #
                # kl *= klw

                # if sample:
                #     distributions = [ContinuousBernoulli(d).rsample()
                #                      # .to(device)
                #                      for d in
                #                      distributions]
            else:
                kl_losses.append(0)

            if sample:
                distributions = ContinuousBernoulli(distributions).rsample()
                # distributions = [ContinuousBernoulli(d).rsample()
                #                  # .to(device)
                #                  for d in
                #                  distributions]

            if recursive:
                assert False

                _pred = []

                y_output = preds[-1]

                for i in range(preds.shape[0] - 1, -1, -1):
                    # for y_b, y_c in zip(preds[-2::-1], distributions[-2::-1]):
                    # gate = torch.sigmoid(y_c)
                    y_b = preds[i]
                    y_c = distributions[i]

                    y_output = y_b * y_c + (1 - y_c) * y_output

                loss = nn.functional.cross_entropy(y_output, y,
                                                   reduction='mean')

            else:
                distributions[:, 1:, :] *= torch.cumprod(
                    1 - distributions[:, :-1, :], 1)

                d = distributions
                # d[:, 1:] *= torch.cumprod(1 - d[:, :-1], 1)

                # distributions = distributions[:, 1:, :] * \
                #                 torch.cumprod(1 - distributions[:, :-1, :], 1)

                # drop = torch.full(distributions.shape[:2], 0.5,
                #                   device=device)
                #
                # mask = torch.bernoulli(drop).unsqueeze(-1)
                # d = mask * distributions

                if joint_type == 'predictions':

                    # a = torch.quantile(distributions, 0.75, 1, keepdim=True)
                    # b = (distributions >= a).float() * distributions
                    # distributions = b

                    preds = preds * d
                    f_hat = torch.amax(preds, 1)
                    # f_hat = torch.mean(preds, 1)

                    # a = torch.topk(distributions, 2, 0)
                    # preds = torch.gather(preds, 0, a[1])
                    #
                    # f_hat = torch.mean(preds, 0)

                    # preds = preds * distributions

                    # output = cos(preds, preds)

                    # distances = []

                    # drop = torch.full(preds.shape, 0.25,
                    #                   device=device)
                    # drop = torch.full(preds.shape, 0.25,
                    #                   device=device)
                    #
                    # mask = torch.bernoulli(drop)
                    # preds = mask * preds

                    # f_hat = torch.max(preds, 0)[0]
                    # f_hat = torch.amax(preds, 0)

                    # f_hat = preds.sum(0)

                    # for i in range(len(preds)):
                    #     d = 0
                    #     for j in range(i, len(preds)):
                    #         similarity = torch.cosine_similarity(
                    #             preds[i].view(1, -1),
                    #             preds[j].view(1, -1)) ** 2
                    #         d += similarity
                    #
                    #     d = d / (len(preds) - i)
                    #     distances.append(d)
                    #
                    # distances = torch.cat(distances)

                    # f_hat = preds.mean(0)
                    loss = nn.functional.cross_entropy(f_hat, y,
                                                       reduction='mean')

                    # loss += distances.mean()

                else:

                    loss = torch.stack(
                        [nn.functional.cross_entropy(preds[:, p], y,
                                                     reduction='none')
                         for p in range(preds.shape[1])], 1)

                    distributions = distributions.squeeze(-1)

                    loss = loss * distributions
                    loss = loss.sum(1)
                    loss = loss.mean()

            losses.append(loss.item())

            loss += kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = np.mean(losses)

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores = standard_eval(model=model,
                                        dataset_loader=eval_loader,
                                        topk=[1, 5],
                                        classifier=predictors[-1])
        else:
            eval_scores = 0

        if early_stopping is not None:
            # r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
            #     else early_stopping.step(mean_loss)
            r = early_stopping.step(mean_loss)
            if r < 0:
                break
            elif r > 0:
                best_model = (model.state_dict(), predictors.state_dict())
                best_model_i = epoch
        else:
            best_model = (model.state_dict(), predictors.state_dict())
            best_model_i = epoch

        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])

        scores.append(test_scores)

        s = branches_eval(model=model,
                          dataset_loader=test_loader,
                          predictors=predictors)
        s = dict(s)
        print(s)

        for epsilon in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]:
            a, b = binary_eval(model=model,
                               dataset_loader=test_loader,
                               predictors=predictors,
                               epsilon=epsilon,
                               cumulative_threshold=True)

            a, b = dict(a), dict(b)

            print('Epsilon {} scores: {}, {}'.format(epsilon,
                                                        dict(a), dict(b)))

        mean_kl_loss = np.mean(kl_losses)
        mean_losses.append(mean_loss)
        bar.set_postfix(
            {
                'Train score': train_scores, 'Test score': test_scores,
                'Eval score': eval_scores if eval_scores != 0 else 0,
                'Mean loss': mean_loss, 'mean kl loss': mean_kl_loss
            })

    return best_model, scores, scores[best_model_i], mean_losses
