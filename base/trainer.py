import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence, Bernoulli, \
    ContinuousBernoulli, RelaxedBernoulli

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

from base.evaluators import standard_eval, branches_eval, binary_eval, \
    binary_statistics
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
                  joint_type='logits',
                  early_stopping=None,
                  test_loader=None, eval_loader=None):
    device = get_device(model)

    if joint_type not in ['losses', 'logits']:
        raise ValueError

    if weights is None:
        weights = torch.tensor([1.0] * model.n_branches(), device=device)

    if not isinstance(weights, (torch.Tensor, torch.nn.Parameter)):
        if isinstance(weights, (int, float)):
            weights = torch.tensor([weights] * model.n_branches(),
                                   device=device, dtype=torch.float)

        else:
            weights = torch.tensor(weights, device=device, dtype=torch.float)

        if joint_type == 'logits':
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

            if joint_type == 'logits':
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
                            joint_type='logits',
                            sample=True,
                            scheduler=None,
                            early_stopping=None,
                            test_loader=None,
                            eval_loader=None,
                            recursive=False,
                            fix_last_layer=False,
                            normalize_weights=True,
                            prior_mode='ones',
                            regularization_loss='bce',
                            calibrate=False):
    device = get_device(model)

    if joint_type not in ['losses', 'logits']:
        raise ValueError

    if prior_mode not in ['entropy', 'ones', 'probability']:
        raise ValueError

    if regularization_loss not in ['kl', 'bce']:
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

    temperature = [max(1, 10 * np.exp(-0.5 * t)) for t in range(epochs)]
    temperature = [1 * np.exp(-0.25 * t) for t in range(epochs)]

    temperature = np.linspace(1, 0.1, epochs)

    print(temperature)

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
                    if prior_mode == 'probability':
                        _y = y.unsqueeze(-1)
                        _y = _y.expand(-1, 5).unsqueeze(-1)
                        sf = torch.softmax(preds, -1)

                        prior_gt = torch.gather(sf, -1, _y)

                        # _y = y.unsqueeze(1)
                        # mx = torch.argmax(preds, -1)
                        #
                        # mask = (mx == _y).float()
                        # prior_gt = prior_gt * mask.unsqueeze(-1)

                    elif prior_mode == 'entropy':
                        sf = torch.softmax(preds, -1)
                        h = -(sf + 1e-12).log() * sf
                        h = h / np.log(sf.shape[-1])
                        h = h.sum(-1)
                        prior_gt = 1 - h

                    elif prior_mode == 'ones':
                        _y = y.unsqueeze(1)
                        mx = torch.argmax(preds, -1)
                        prior_gt = (mx == _y).float()
                        prior_gt = prior_gt.unsqueeze(-1)

                    if fix_last_layer:
                        prior_gt[:, -1] = 1
                        # d = distributions[:, :-1]
                    # else:
                    #     d = distributions

                    # d = distributions[:, :-1]
                # else:
                #     d = distributions

                # bce = nn.functional.binary_cross_entropy(
                #     distributions[:, :-1], a, reduction='none')

                if regularization_loss == 'bce':

                    bce = nn.functional.binary_cross_entropy(distributions,
                                                             prior_gt,
                                                             reduction='none')
                    kl = bce.sum(1)
                    # kl = bce

                elif regularization_loss == 'kl':
                    # kl = bce.sum([-1, -2])

                    # a = torch.maximum(a, prior_stack)
                    d = ContinuousBernoulli(distributions)
                    p = ContinuousBernoulli(prior_gt)
                    kl = kl_divergence(d, p).sum(1)

                kl = kl.squeeze()
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
                # # # klw = 2 ** (epochs - epoch - 1) \
                # # #       / (2 ** epochs - 1)
                # # #
                # #
                # #
                # kl *= (1 - klw)

                # if sample:
                #     distributions = [ContinuousBernoulli(d).rsample()
                #                      # .to(device)
                #                      for d in
                #                      distributions]
            else:
                kl_losses.append(0)

            if sample:
                distributions = RelaxedBernoulli(1,
                                                 distributions).rsample()

                if fix_last_layer:
                    distributions[:, -1] = 1

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
                # distributions[:, 1:, :] = distributions[:, 1:,
                #                           :] * torch.cumprod(
                #     1 - distributions[:, :-1, :], 1)

                # drop = torch.full(distributions.shape[:2], 0.5,
                #                   device=device)
                #
                # mask = torch.bernoulli(drop).unsqueeze(-1)
                # distributions = mask * distributions
                #
                # if fix_last_layer:
                #     distributions[:, -1] = 1

                # distributions = 1 - distributions

                if normalize_weights:
                    a, b = torch.split(distributions,
                                              [distributions.shape[1] - 1, 1],
                                              dim=1)

                    c = torch.cumprod(1 - a, 1)
                    # c1 = torch.sum(c, 1, keepdim=True)

                    cat = torch.cat((torch.ones_like(b), c), 1)
                    distributions = distributions * cat
                    # distributions = torch.cat((prior_gt, c), 1)

                # torch.stack((d[:, 1, :], d[:, 1:, :]))
                #
                # d[:, 1:, :] = d[:, 1:, :] * torch.cumprod(1 - d[:, :-1, :], 1)

                # d[:, 1:] *= torch.cumprod(1 - d[:, :-1], 1)

                # distributions = distributions[:, 1:, :] * \
                #                 torch.cumprod(1 - distributions[:, :-1, :], 1)

                # drop = torch.full(distributions.shape[:2], 0.5,
                #                   device=device)
                #
                # mask = torch.bernoulli(drop).unsqueeze(-1)
                # distributions = mask * distributions

                if joint_type == 'logits':

                    # a = torch.quantile(distributions, 0.75, 1, keepdim=True)
                    # b = (distributions >= a).float() * distributions
                    # distributions = b

                    preds = preds * distributions
                    # f_hat = torch.amax(preds, 1)
                    f_hat = torch.sum(preds, 1)

                    # f_hat = torch.mean(preds, 1)
                    # f_hat = f_hat / torch.sum(distributions, 1)

                    loss = nn.functional.cross_entropy(f_hat, y,
                                                       reduction='mean')

                    # loss += distances.mean()

                else:
                    # distributions = 1 - distributions

                    loss = torch.stack(
                        [nn.functional.cross_entropy(preds[:, p], y,
                                                     reduction='none')
                         for p in range(preds.shape[1])], 1)

                    distributions = distributions.squeeze(-1)

                    loss = loss * distributions
                    loss = loss.mean(0)
                    loss = loss.sum()

            losses.append(loss.item())

            # loss += kl

            w = temperature[epoch]
            # loss = w * loss + (1 - w) * kl

            loss = loss + kl

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

        print(binary_statistics(model=model,
                                dataset_loader=test_loader,
                                predictors=predictors))

        for epsilon in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            prior_gt, b = binary_eval(model=model,
                                      dataset_loader=test_loader,
                                      predictors=predictors,
                                      epsilon=[0.7 if epsilon <= 0.7 else epsilon] +
                                           [epsilon] *
                                           (model.n_branches() - 1),
                                      cumulative_threshold=False)

            prior_gt, b = dict(prior_gt), dict(b)

            s = '\tEpsilon {}. '.format(epsilon)
            for k in sorted([k for k in prior_gt.keys() if k != 'global']):
                s += 'Branch {}, score: {}, counter: {}. '.format(k,
                                                                  prior_gt[k],
                                                                  b[k])
            s += 'Global score: {}'.format(prior_gt['global'])

            print(s)
            # print('Epsilon {} scores: {}, {}'.format(epsilon,
            #                                          dict(prior_gt), dict(b)))

        mean_kl_loss = np.mean(kl_losses)
        mean_losses.append(mean_loss)
        bar.set_postfix(
            {
                'w': w,
                'Train score': train_scores, 'Test score': test_scores,
                'Eval score': eval_scores if eval_scores != 0 else 0,
                'Mean loss': mean_loss, 'mean kl loss': mean_kl_loss
            })

    return best_model, scores, scores[best_model_i], mean_losses
