import os
import pickle
import sys
from builtins import print
from collections import defaultdict
from itertools import chain
from typing import Union

import numpy as np
import torch
import torch.nn as nn
# from experiments.calibration import ece_score
# from eval import eval_method
# from experiments.corrupted_cifar import corrupted_cifar_uncertainty
# from experiments.fgsm import perturbed_predictions
# from methods import SingleModel, Naive, SuperMask
# from methods.batch_ensemble.batch_ensemble import BatchEnsemble
# from methods.dropout.dropout import MCDropout
# from methods.snapshot.snapshot import Snapshot
# from methods.supermask.supermask import ExtremeBatchPruningSuperMask
# from utils import get_optimizer, get_dataset, get_model, EarlyStopping, \
#     ensures_path, calculate_trainable_parameters
import yaml
import logging
import argparse

from torch.distributions import ContinuousBernoulli

from base.evaluators import branches_entropy, branches_eval, standard_eval, \
    branches_binary, eval_branches_entropy
from base.trainer import joint_trainer, standard_trainer, \
    output_combiner_trainer, binary_classifier_trainer, bayesian_joint_trainer, \
    posterior_classifier_trainer
from base.utils import get_optimizer, EarlyStopping, get_dataset, get_model
from bayesian.posteriors import BayesianHead, BayesianHeads, \
    LayerEmbeddingContBernoulli


def ensures_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='Paths of the yaml con figuration files')

parser.add_argument('--device',
                    required=False,
                    default=None,
                    type=int,
                    help='')

args = parser.parse_args()

for experiment in args.files:

    with open(experiment, 'r') as stream:
        experiment_config = yaml.safe_load(stream)

    to_save = experiment_config.get('save', True)
    to_load = experiment_config.get('load', True)

    save_path = experiment_config['save_path']
    experiment_path = str(experiment_config['name'])

    ensures_path(save_path)

    with open(experiment_config['optimizer'], 'r') as file:
        optimizer_config = yaml.safe_load(file)
        optimizer_config = dict(optimizer_config)

    optimizer, regularization, scheduler = get_optimizer(**optimizer_config)

    if args.device is not None:
        device = args.device
    else:
        device = experiment_config.get('device', 'cpu')

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    device = torch.device(device)

    with open(experiment_config['trainer'], 'r') as file:
        trainer = yaml.safe_load(file)
        trainer = dict(trainer)
    eval_percentage = trainer.get('eval', None)

    if 'early_stopping' in trainer:
        early_stopping_dict = dict(trainer['early_stopping'])
        early_stopping_value = early_stopping_dict.get('value', 'loss')
        assert early_stopping_value in ['eval', 'loss']
        if early_stopping_value == 'eval':
            assert eval_percentage is not None and eval_percentage > 0
        early_stopping = EarlyStopping(
            tolerance=early_stopping_dict['tolerance'],
            min=early_stopping_value == 'loss')
    else:
        early_stopping = None
        early_stopping_value = None

    batch_size = trainer['batch_size']
    epochs = trainer['epochs']
    experiments = trainer.get('experiments', 1)

    train, test, eval, input_size, classes = get_dataset(trainer['dataset'],
                                                         model_name=trainer[
                                                             'model'],
                                                         augment=False)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True, num_workers=4)
    eval_loader = None

    for experiment_seed in range(experiments):
        np.random.seed(experiment_seed)
        torch.random.manual_seed(experiment_seed)

        seed_path = os.path.join(save_path, experiment_path,
                                 str(experiment_seed))

        already_present = ensures_path(seed_path)

        hs = [
            logging.StreamHandler(sys.stdout),
            # logging.StreamHandler(sys.stderr)
        ]

        hs.append(
            logging.FileHandler(os.path.join(seed_path, 'info.log'), mode='w'))

        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=hs
                            )

        logger = logging.getLogger(__name__)

        config_info = experiment_config.copy()
        config_info.update({'optimizer': optimizer_config, 'trainer': trainer})

        logger.info('Config file \n{}'.format(
            yaml.dump(config_info, allow_unicode=True,
                      default_flow_style=False)))

        logger.info('Experiment {}/{}'.format(experiment_seed + 1, experiments))

        eval_loader = None
        if eval is not None:
            eval_loader = torch.utils.data.DataLoader(dataset=eval,
                                                      batch_size=batch_size,
                                                      shuffle=False)
        elif eval_percentage is not None and eval_percentage > 0:
            assert eval_percentage < 1
            train_len = len(train)
            eval_len = int(train_len * eval_percentage)
            train_len = train_len - eval_len

            train, eval = torch.utils.data.random_split(train,
                                                        [train_len, eval_len])
            train_loader = torch.utils.data.DataLoader(dataset=train,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            eval_loader = torch.utils.data.DataLoader(dataset=eval,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            logger.info('Train dataset size: {}'.format(len(train)))
            logger.info('Test dataset size: {}'.format(len(test)))
            logger.info(
                'Eval dataset created, having size: {}'.format(len(eval)))

        method_name = experiment_config.get('method', None).lower()

        method_parameters = dict(experiment_config.get('method_parameters', {}))

        model = get_model(name=trainer['model'], input_size=input_size,
                          output=classes)
        model.to(device)

        predictors = None
        binaries = None

        if experiment_config.get('fine_tuning', False):

            ensures_path('results/models')

            p = os.path.join('results/models',
                             '{}_{}.pt'.format(trainer['model'].lower(),
                                               experiment_seed))

            if to_load and os.path.exists(p):
                logger.info('Base model loaded.')

                model_state_dict = torch.load(p, map_location=device)
                model.load_state_dict(model_state_dict)

            else:
                logger.info('Base model training.')

                with open(experiment_config['fine_tuning_trainer'],
                          'r') as file:
                    fine_tune_trainer = yaml.safe_load(file)

                with open(experiment_config['fine_tuning_optimizer'],
                          'r') as file:
                    fine_tune_optimizer_config = yaml.safe_load(file)

                optimizer, regularization, scheduler = get_optimizer(
                    **fine_tune_optimizer_config)

                opt = optimizer(model.parameters())

                best_base_model, _, _, _ = standard_trainer(model=model,
                                                            optimizer=opt,
                                                            train_loader=train_loader,
                                                            epochs=
                                                            fine_tune_trainer[
                                                                'epochs'],
                                                            scheduler=None,
                                                            early_stopping=early_stopping,
                                                            test_loader=test_loader,
                                                            eval_loader=eval_loader,
                                                            device=device)
                model.load_state_dict(best_base_model)
                torch.save(best_base_model, p)

            s = standard_eval(model, test_loader, topk=[1, 5],
                              device=device)
            logger.info('Base model scores: {}'.format(s))

        if method_name == 'joint' or method_name == 'combined_output':
            predictors = nn.ModuleList()
            predictors.to(device)

            x, y = next(iter(train_loader))
            _, outputs = model(x.to(device))

            for o in outputs:
                inc = o.shape[1]
                bf = torch.flatten(o, 1)
                cv = nn.Conv2d(inc, 128, kernel_size=3).to(device)
                f = torch.flatten(cv(o), 1)

                predictors.append(
                    nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
                                  nn.Flatten(),
                                  nn.Linear(f.shape[-1], classes)))

            predictors.to(device)

            if to_load and os.path.exists(os.path.join(seed_path, 'model.pt')):
                # model_state_dict = torch.load(seed_path)
                # model.load_state_dict(model_state_dict)

                md = torch.load(os.path.join(seed_path, 'model.pt'),
                                map_location=device)
                pd = torch.load(os.path.join(seed_path, 'predictors.pt'),
                                map_location=device)

                model.load_state_dict(md)
                predictors.load_state_dict(pd)

            else:

                opt = optimizer(chain(model.parameters(),
                                      predictors.parameters()))

                if method_name == 'joint':
                    (best_model, best_predictors), _, _, _ = joint_trainer(
                        model=model,
                        predictors=predictors,
                        optimizer=opt,
                        train_loader=train_loader,
                        epochs=epochs,
                        scheduler=None,
                        early_stopping=early_stopping,
                        test_loader=test_loader,
                        eval_loader=eval_loader,
                        device=device)
                elif method_name == 'combined_output':
                    weights = experiment_config.get('weights')
                    convex_combination = experiment_config. \
                        get('convex_combination', False)

                    (best_model, best_predictors), _, _, _ = \
                        output_combiner_trainer(
                            model=model,
                            predictors=predictors,
                            optimizer=opt,
                            train_loader=train_loader,
                            epochs=epochs,
                            scheduler=None,
                            early_stopping=early_stopping,
                            weights=weights,
                            convex_combination=convex_combination,
                            test_loader=test_loader,
                            eval_loader=eval_loader,
                            device=device)
                else:
                    assert False

                model.load_state_dict(best_model)
                predictors.load_state_dict(best_predictors)

                torch.save(best_model, os.path.join(seed_path, 'model.pt'))
                torch.save(best_predictors,
                           os.path.join(seed_path, 'predictors.pt'))

        elif method_name == 'binary':
            predictors = nn.ModuleList()
            binaries = nn.ModuleList()

            x, y = next(iter(train_loader))
            _, outputs = model(x.to(device))

            for o in outputs:
                inc = o.shape[1]
                bf = torch.flatten(o, 1)

                cv = nn.Conv2d(inc, 128, kernel_size=3).to(device)
                cv1 = nn.MaxPool2d(kernel_size=2).to(device)
                f = torch.flatten(cv1(cv(o)), 1)

                predictors.append(
                    nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Dropout(0.5),
                                  nn.Flatten(),
                                  nn.Linear(f.shape[-1], classes)))
                binaries.append(
                    nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Dropout(0.5),
                                  nn.Flatten(),
                                  nn.Linear(f.shape[-1], 1),
                                  nn.Sigmoid()))

            predictors.to(device)
            binaries.to(device)

            if to_load and os.path.exists(os.path.join(seed_path, 'model.pt')):
                # model_state_dict = torch.load(seed_path)
                # model.load_state_dict(model_state_dict)

                md = torch.load(os.path.join(seed_path, 'model.pt'),
                                map_location=device)
                pd = torch.load(os.path.join(seed_path, 'predictors.pt'),
                                map_location=device)
                bd = torch.load(os.path.join(seed_path,
                                             'binary_classifiers.pt'),
                                map_location=device)

                model.load_state_dict(md)
                predictors.load_state_dict(pd)
                binaries.load_state_dict(bd)
            else:

                opt = optimizer(chain(model.parameters(),
                                      predictors.parameters(),
                                      binaries.parameters()))

                # if method_name == 'joint':
                (best_model, best_predictors,
                 best_binaries), _, _, _ = binary_classifier_trainer(
                    model=model,
                    predictors=predictors,
                    binary_classifiers=binaries,
                    optimizer=opt,
                    train_loader=train_loader,
                    epochs=epochs,
                    scheduler=None,
                    energy_w=experiment_config.get('energy_w', 1e-8),
                    early_stopping=early_stopping,
                    test_loader=test_loader,
                    eval_loader=eval_loader,
                    device=device)

                # elif method_name == 'combined_output':
                #     weights = experiment_config.get('weights')
                #     convex_combination = experiment_config. \
                #         get('convex_combination', False)
                #
                #     (best_model, best_predictors), _, _, _ = \
                #         output_combiner_trainer(
                #             model=model,
                #             predictors=predictors,
                #             optimizer=opt,
                #             train_loader=train_loader,
                #             epochs=epochs,
                #             scheduler=None,
                #             early_stopping=None,
                #             weights=weights,
                #             convex_combination=convex_combination,
                #             test_loader=test_loader,
                #             eval_loader=eval_loader,
                #             device=device)
                # else:
                #     assert False

                model.load_state_dict(best_model)
                predictors.load_state_dict(best_predictors)
                binaries.load_state_dict(best_binaries)

                torch.save(best_model, os.path.join(seed_path, 'model.pt'))
                torch.save(best_predictors,
                           os.path.join(seed_path, 'predictors.pt'))
                torch.save(best_binaries,
                           os.path.join(seed_path, 'binary_classifiers.pt'))

        elif method_name == 'bayesian_joint':
            # predictors = nn.ModuleList()
            # predictors.to(device)

            x, y = next(iter(train_loader))
            _, outputs = model(x.to(device))

            heads = []

            for o in outputs:
                h = BayesianHead(o, classes, device)
                heads.append(h)

            predictors = BayesianHeads(heads)

            predictors.to(device)

            if to_load and os.path.exists(os.path.join(seed_path, 'model.pt')):
                # model_state_dict = torch.load(seed_path)
                # model.load_state_dict(model_state_dict)

                md = torch.load(os.path.join(seed_path, 'model.pt'),
                                map_location=device)
                pd = torch.load(os.path.join(seed_path, 'predictors.pt'),
                                map_location=device)

                model.load_state_dict(md)
                predictors.load_state_dict(pd)

            else:

                opt = optimizer(chain(model.parameters(),
                                      predictors.parameters()))

                if method_name == 'bayesian_joint':
                    (best_model,
                     best_predictors), _, _, _ = bayesian_joint_trainer(
                        model=model,
                        predictors=predictors,
                        optimizer=opt,
                        train_loader=train_loader,
                        epochs=epochs,
                        scheduler=None,
                        early_stopping=early_stopping,
                        samples=experiment_config.get('train_samples', 1),
                        test_loader=test_loader,
                        eval_loader=eval_loader,
                        prior_w=experiment_config.get('prior_w', 1),
                        device=device)

                # elif method_name == 'combined_output':
                #     weights = experiment_config.get('weights')
                #     convex_combination = experiment_config. \
                #         get('convex_combination', False)
                #
                #     (best_model, best_predictors), _, _, _ = \
                #         output_combiner_trainer(
                #             model=model,
                #             predictors=predictors,
                #             optimizer=opt,
                #             train_loader=train_loader,
                #             epochs=epochs,
                #             scheduler=None,
                #             early_stopping=None,
                #             weights=weights,
                #             convex_combination=convex_combination,
                #             test_loader=test_loader,
                #             eval_loader=eval_loader,
                #             device=device)
                else:
                    assert False

                model.load_state_dict(best_model)
                predictors.load_state_dict(best_predictors)

                torch.save(best_model, os.path.join(seed_path, 'model.pt'))
                torch.save(best_predictors,
                           os.path.join(seed_path, 'predictors.pt'))

        # elif method_name == 'bayesian_binary':
        #     predictors = nn.ModuleList()
        #     binaries = nn.ModuleList()
        #
        #     x, y = next(iter(train_loader))
        #     _, outputs = model(x.to(device))
        #
        #     for o in outputs:
        #         inc = o.shape[1]
        #         bf = torch.flatten(o, 1)
        #         cv = nn.Conv2d(inc, 128, kernel_size=3).to(device)
        #         f = torch.flatten(cv(o), 1)
        #
        #         predictors.append(
        #             nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
        #                           nn.Flatten(),
        #                           nn.Linear(f.shape[-1], classes)))
        #         binaries.append(
        #             nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
        #                           nn.Flatten(),
        #                           nn.Linear(f.shape[-1], 1),
        #                           nn.Sigmoid()))
        #
        #     posteriors = LayerEmbeddingContBernoulli(alpha_layers=binaries)
        #
        #     predictors.to(device)
        #     posteriors.to(device)
        #
        #     if to_load and os.path.exists(os.path.join(seed_path, 'model.pt')):
        #         # model_state_dict = torch.load(seed_path)
        #         # model.load_state_dict(model_state_dict)
        #
        #         md = torch.load(os.path.join(seed_path, 'model.pt'),
        #                         map_location=device)
        #         pd = torch.load(os.path.join(seed_path, 'predictors.pt'),
        #                         map_location=device)
        #         bd = torch.load(os.path.join(seed_path,
        #                                      'posteriors.pt'),
        #                         map_location=device)
        #
        #         model.load_state_dict(md)
        #         predictors.load_state_dict(pd)
        #         posteriors.load_state_dict(bd)
        #     else:
        #
        #         opt = optimizer(chain(model.parameters(),
        #                               predictors.parameters(),
        #                               posteriors.parameters()))
        #
        #         prior = ContinuousBernoulli(torch.tensor([0.2], device=device))
        #
        #         # if method_name == 'joint':
        #         (best_model, best_predictors,
        #          best_binaries), _, _, _ = posterior_classifier_trainer(
        #             model=model,
        #             predictors=predictors,
        #             posteriors=posteriors,
        #             optimizer=opt,
        #             prior=prior,
        #             prior_w=experiment_config.get('prior_w', 0.1),
        #             entropy_w=experiment_config.get('entropy_w', 1),
        #             train_loader=train_loader,
        #             epochs=epochs,
        #             use_mmd=experiment_config.get('mmd', False),
        #             scheduler=None,
        #             early_stopping=early_stopping,
        #             test_loader=test_loader,
        #             eval_loader=eval_loader,
        #             device=device,
        #             cumulative_prior=False)
        #
        #         # elif method_name == 'combined_output':
        #         #     weights = experiment_config.get('weights')
        #         #     convex_combination = experiment_config. \
        #         #         get('convex_combination', False)
        #         #
        #         #     (best_model, best_predictors), _, _, _ = \
        #         #         output_combiner_trainer(
        #         #             model=model,
        #         #             predictors=predictors,
        #         #             optimizer=opt,
        #         #             train_loader=train_loader,
        #         #             epochs=epochs,
        #         #             scheduler=None,
        #         #             early_stopping=None,
        #         #             weights=weights,
        #         #             convex_combination=convex_combination,
        #         #             test_loader=test_loader,
        #         #             eval_loader=eval_loader,
        #         #             device=device)
        #         # else:
        #         #     assert False
        #
        #         model.load_state_dict(best_model)
        #         predictors.load_state_dict(best_predictors)
        #         posteriors.load_state_dict(best_binaries)
        #
        #         torch.save(best_model, os.path.join(seed_path, 'model.pt'))
        #         torch.save(best_predictors,
        #                    os.path.join(seed_path, 'predictors.pt'))
        #         torch.save(best_binaries,
        #                    os.path.join(seed_path, 'posteriors.pt'))
        else:
            assert False, 'Accepted methods: joint, combined_output, binary'

        if method_name in ['joint',
                           'combined_output',
                           'binary',
                           'bayesian_joint',
                           'bayesian_binary']:

            s = branches_eval(model, predictors, test_loader, device=device)
            logger.info('Branches scores {}'.format(s))

            logger.info('Percentile entropy experiment')

            ps = experiment_config.get('entropy_threshold',
                                       [0.1, 0.15,
                                        0.20, 0.25,
                                        0.30, 0.5, 0.75])
            if isinstance(ps, float):
                ps = [ps]

            for p in ps:
                s, c = eval_branches_entropy(model,
                                             predictors,
                                             percentile=p,
                                             dataset_loader=test_loader,
                                             eval_loader=eval_loader,
                                             device=device)

                logger.info('Percentile: {}'.format(p))
                logger.info('\tScores: {}'.format(s))
                logger.info('\tCounters: {}'.format(c))

            logger.info('Branches entropy experiment')
            th = experiment_config.get('entropy_threshold',
                                       [0.001, 0.01,
                                        0.05, 0.1,
                                        0.25, 0.5, 0.75])
            if isinstance(th, float):
                th = [th]

            for h in th:
                s, c = branches_entropy(model=model,
                                        threshold=h,
                                        dataset_loader=test_loader,
                                        predictors=predictors,
                                        device=device,
                                        samples=
                                        experiment_config.get('test_samples',
                                                              1))

                logger.info('Threshold: {}'.format(h))
                logger.info('\tCumulative score: {}'.format(s['global']))
                logger.info('\tScores: {}'.format(s))
                logger.info('\tCounters: {}'.format(c))

        if method_name == 'binary' or method_name == 'bayesian_binary':
            th = experiment_config.get('binary_threshold', 0.5)

            # s = branches_eval(model, predictors, test_loader, device=device)
            # logger.info('Branches scores {}'.format(s))

            logger.info('Branches binary experiment')
            s, c = branches_binary(model=model,
                                   binary_classifiers=binaries,
                                   dataset_loader=test_loader,
                                   predictors=predictors,
                                   threshold=th,
                                   device=device)

            logger.info('\tScores: {}'.format(s))
            logger.info('\tCounters: {}'.format(c))

        # if method_name is None or method_name == 'normal':
        #     method_name = 'normal'
        #     method = SingleModel(model=model, device=device)
        # elif method_name == 'naive':
        #     method = Naive(model=model, device=device,
        #                    method_parameters=method_parameters)
        # elif method_name == 'supermask':
        #     method = SuperMask(model=model, method_parameters=method_parameters,
        #                        device=device)
        # elif method_name == 'batch_ensemble':
        #     method = BatchEnsemble(model=model,
        #                            method_parameters=method_parameters,
        #                            device=device)
        # elif method_name == 'batch_supermask':
        #     method = ExtremeBatchPruningSuperMask(model=model,
        #                                           method_parameters=method_parameters,
        #                                           device=device)
        # elif method_name == 'mc_dropout':
        #     method = MCDropout(model=model, method_parameters=method_parameters,
        #                        device=device)
        # elif method_name == 'snapshot':
        #     method = Snapshot(model=model, method_parameters=method_parameters,
        #                       device=device)
        # else:
        #     assert False

        # logger.info('Method used: {}'.format(method_name))

        # if to_load and os.path.exists(os.path.join(seed_path, 'results.pkl')):
        #     method.load(os.path.join(seed_path))
        #     with open(os.path.join(seed_path, 'results.pkl'), 'rb') as file:
        #         results = pickle.load(file)
        #     logger.info('Results and models loaded.')
        # else:
        #     results = method.train_models(optimizer=optimizer,
        #                                   train_dataset=train_loader,
        #                                   epochs=epochs,
        #                                   scheduler=scheduler,
        #                                   early_stopping=early_stopping,
        #                                   test_dataset=test_loader,
        #                                   eval_dataset=eval_loader)
        #
        #     if to_save:
        #         method.save(seed_path)
        #         with open(os.path.join(seed_path, 'results.pkl'), 'wb') as file:
        #             pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        #         logger.info('Results and models saved.')

        # logger.info('Ensemble '
        #             'score on test: {}'.format(
        #     eval_method(method, dataset=test_loader)[0]))

        # params = 0
        # if hasattr(method, 'models'):
        #     models = method.models
        # else:
        #     models = [method.model]
        #
        # for m in models:
        #     params += calculate_trainable_parameters(m)
        #
        # logger.info('Method {} has {} parameters'.format(method_name, params))
        #
        # ece, _, _, _ = ece_score(method, test_loader)
        #
        # logger.info('Ece score: {}'.format(ece))
