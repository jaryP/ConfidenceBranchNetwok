import json
import logging
import os
import warnings
from itertools import chain

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from base.evaluators import binary_eval, entropy_eval, standard_eval, \
    branches_eval
from base.trainer import binary_bernulli_trainer, joint_trainer, \
    standard_trainer
from utils import get_dataset, get_optimizer, get_model


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    model_cfg = cfg['model']
    model_name = model_cfg['name']

    dataset_cfg = cfg['dataset']
    dataset_name = dataset_cfg['name']
    augmented_dataset = dataset_cfg.get('augment', False)

    experiment_cfg = cfg['experiment']
    load, save, path, experiments = experiment_cfg.get('load', True), \
                                    experiment_cfg.get('save', True), \
                                    experiment_cfg.get('path', None), \
                                    experiment_cfg.get('experiments', 1)

    plot = experiment_cfg.get('plot', False)

    method_cfg = cfg['method']
    method_name = method_cfg['name']

    get_binaries = True if method_name == 'bernulli' else False

    # trainer = None

    # if method_name == 'bernulli':
    #     trainer = binary_bernulli_trainer

    # distance_regularization, distance_weight, similarity = method_cfg.get(
    #     'distance_regularization', True), \
    #                                                        method_cfg.get(
    #                                                            'distance_weight',
    #                                                            1), \
    #                                                        method_cfg[
    #                                                            'similarity']
    # ensemble_dropout = method_cfg.get('ensemble_dropout', 0)
    # anneal_dirichlet = method_cfg.get('anneal_dirichlet', True)
    # test_samples = method_cfg.get('test_samples', 1)

    training_cfg = cfg['training']
    epochs, batch_size, device = training_cfg['epochs'], \
                                 training_cfg['batch_size'], \
                                 training_cfg.get('device', 'cpu')
    eval_percentage = training_cfg.get('eval_split', None)

    optimizer_cfg = cfg['optimizer']
    optimizer_name, lr, momentum, weight_decay = optimizer_cfg.get('optimizer',
                                                                   'sgd'), \
                                                 optimizer_cfg.get('lr', 1e-1), \
                                                 optimizer_cfg.get('momentum',
                                                                   0.9), \
                                                 optimizer_cfg.get(
                                                     'weight_decay', 0)

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn("Device not found or CUDA not available.")

    device = torch.device(device)

    if not isinstance(experiments, int):
        raise ValueError('experiments argument must be integer: {} given.'
                         .format(experiments))

    if path is None:
        path = os.getcwd()
    else:
        os.chdir(path)
        os.makedirs(path, exist_ok=True)

    for experiment in range(experiments):
        log.info('Experiment #{}'.format(experiment))

        torch.manual_seed(experiment)
        np.random.seed(experiment)

        experiment_path = os.path.join(path, 'exp_{}'.format(experiment))
        os.makedirs(experiment_path, exist_ok=True)

        eval_loader = None
        train_set, test_set, input_size, classes = \
            get_dataset(name=dataset_name,
                        model_name=None,
                        augmentation=augmented_dataset)

        if eval_percentage is not None and eval_percentage > 0:
            assert eval_percentage < 1
            train_len = len(train_set)
            eval_len = int(train_len * eval_percentage)
            train_len = train_len - eval_len

            train_set, eval = torch.utils.data.random_split(train_set,
                                                        [train_len, eval_len])
            # train_loader = torch.utils.data.DataLoader(dataset=train,
            #                                            batch_size=batch_size,
            #                                            shuffle=True)
            eval_loader = torch.utils.data.DataLoader(dataset=eval,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            # logger.info('Train dataset size: {}'.format(len(train)))
            # logger.info('Test dataset size: {}'.format(len(test)))
            # logger.info(
            #     'Eval dataset created, having size: {}'.format(len(eval)))


        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        backbone, classifiers = get_model(model_name, image_size=input_size,
                                          classes=classes,
                                          get_binaries=get_binaries)

        if os.path.exists(os.path.join(experiment_path, 'bb.pt')) and load:
            log.info('Model loaded')

            backbone.to(device)
            classifiers.to(device)

            backbone.load_state_dict(torch.load(
                os.path.join(experiment_path, 'bb.pt'), map_location=device))
            classifiers.load_state_dict(torch.load(
                os.path.join(experiment_path, 'classifiers.pt'),
                map_location=device))
        else:
            if method_cfg.get('pre_trained', False):
                pre_trained_path = os.path.join('~/branch_models/',
                                                '{}'.format(dataset_name),
                                                '{}'.format(model_name))

                pre_trained_path = os.path.expanduser(pre_trained_path)

                pre_trained_model_path = os.path.join(pre_trained_path,
                                                      'b{}.pt'.format(
                                                          experiment))

                pre_trained_classifier_path = os.path.join(pre_trained_path,
                                                           'c{}.pt'.format(
                                                               experiment))
                log.info('Pre trained model path {}'.
                         format(pre_trained_path))

                pretrained_backbone, pretrained_classifiers = get_model(
                    model_name,
                    image_size=input_size,
                    classes=classes,
                    get_binaries=get_binaries)

                if os.path.exists(pre_trained_model_path) and \
                        os.path.exists(pre_trained_classifier_path):
                    log.info('Pre trained model loaded')

                    pretrained_backbone.load_state_dict(
                        torch.load(pre_trained_model_path,
                                   map_location=device))
                    pretrained_classifiers.load_state_dict(
                        torch.load(pre_trained_classifier_path,
                                   map_location=device))
                else:

                    os.makedirs(pre_trained_path, exist_ok=True)

                    log.info('Training the base model')

                    pretrained_backbone.to(device)
                    pretrained_classifiers.to(device)

                    parameters = chain(pretrained_backbone.parameters(),
                                       pretrained_classifiers.parameters())

                    optimizer = get_optimizer(parameters=parameters,
                                              name='sgd',
                                              lr=0.01,
                                              momentum=0.9,
                                              weight_decay=0)

                    res = standard_trainer(model=pretrained_backbone,
                                           predictors=pretrained_classifiers,
                                           optimizer=optimizer,
                                           train_loader=trainloader,
                                           epochs=epochs,
                                           scheduler=None,
                                           early_stopping=None,
                                           test_loader=testloader,
                                           eval_loader=eval_loader)[0]

                    backbone_dict, classifiers_dict = res
                    # classifiers.load_state_dict(classifiers_dict)
                    torch.save(backbone_dict,
                               pre_trained_model_path)
                    torch.save(classifiers_dict,
                               pre_trained_classifier_path)

                    pretrained_classifiers.load_state_dict(classifiers_dict)
                    pretrained_backbone.load_state_dict(backbone_dict)

                    log.info('Pre trained model Saved.')

                # train_scores = standard_eval(model=pretrained_backbone,
                #                              dataset_loader=trainloader,
                #                              classifier=pretrained_classifiers[
                #                                  -1])
                #
                # test_scores = standard_eval(model=pretrained_backbone,
                #                             dataset_loader=testloader,
                #                             classifier=pretrained_classifiers[
                #                                 -1])
                #
                # log.info('Pre trained model scores : {}, {}'.format(
                #     train_scores,
                #     test_scores))

                backbone.load_state_dict(pretrained_backbone.state_dict())

            backbone.to(device)
            classifiers.to(device)

            # optimizer = optim.SGD(chain(backbone1.parameters(),
            #                             backbone2.parameters(),
            #                             backbone3.parameters(),
            #                             classifier.parameters()), lr=0.01,
            #                       momentum=0.9)
            log.info('Training started.')

            parameters = chain(backbone.parameters(),
                               classifiers.parameters())

            optimizer = get_optimizer(parameters=parameters,
                                      name=optimizer_name,
                                      lr=lr,
                                      momentum=momentum,
                                      weight_decay=weight_decay)

            if method_name == 'bernulli':

                priors = method_cfg.get('priors', 0.5)
                fixed_bernulli = method_cfg.get('fixed_bernulli', False)
                joint_type = method_cfg.get('joint_type', 'predictions')
                prior_w = method_cfg.get('prior_w', 1e-3)
                sample = method_cfg.get('sample', True)
                recursive = method_cfg.get('recursive', False)

                res = binary_bernulli_trainer(model=backbone,
                                              predictors=classifiers,
                                              bernulli_models=classifiers,
                                              optimizer=optimizer,
                                              train_loader=trainloader,
                                              epochs=epochs,
                                              prior_parameters=priors,
                                              joint_type=joint_type,
                                              prior_w=prior_w,
                                              sample=sample,
                                              eval_loader=eval_loader,
                                              recursive=recursive,
                                              test_loader=testloader,
                                              fixed_bernulli=fixed_bernulli)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'joint':
                joint_type = method_cfg.get('joint_type', 'losses')
                weights = method_cfg.get('weights', None)
                train_weights = method_cfg.get('train_weights', False)

                if train_weights:
                    weights = torch.tensor(weights, device=device,
                                           dtype=torch.float)

                    if joint_type == 'predictions':
                        weights = weights.unsqueeze(-1)
                        weights = weights.unsqueeze(-1)

                    weights = torch.nn.Parameter(weights)
                    parameters = chain(backbone.parameters(),
                                       classifiers.parameters(),
                                       [weights])

                    optimizer = get_optimizer(parameters=parameters,
                                              name=optimizer_name,
                                              lr=lr,
                                              momentum=momentum,
                                              weight_decay=weight_decay)

                res = joint_trainer(model=backbone, predictors=classifiers,
                                    optimizer=optimizer,
                                    weights=weights, train_loader=trainloader,
                                    epochs=epochs,
                                    scheduler=None, joint_type=joint_type,
                                    early_stopping=None, test_loader=testloader,
                                    eval_loader=None)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'standard':

                res = standard_trainer(model=backbone,
                                       predictors=classifiers,
                                       optimizer=optimizer,
                                       train_loader=trainloader,
                                       epochs=epochs,
                                       scheduler=None,
                                       early_stopping=None,
                                       test_loader=testloader,
                                       eval_loader=eval_loader)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            else:
                assert False

            # elif method_name == 'joint':
            #     pass

            #     if logits_training:
            #         backbone, classifiers = dirichlet_logits_model_train(
            #             backbone=backbone,
            #             classifiers=classifiers,
            #             trainloader=trainloader,
            #             testloader=testloader,
            #             optimizer=optimizer,
            #             distance_regularization=distance_regularization,
            #             distance_weight=distance_weight,
            #             similarity=similarity,
            #             epochs=epochs,
            #             anneal_dirichlet=anneal_dirichlet,
            #             ensemble_dropout=ensemble_dropout,
            #             test_samples=test_samples,
            #             device=device)
            #     else:
            #         backbone, classifiers = dirichlet_model_train(
            #             backbone=backbone,
            #             classifiers=classifiers,
            #             trainloader=trainloader,
            #             testloader=testloader,
            #             optimizer=optimizer,
            #             distance_regularization=distance_regularization,
            #             distance_weight=distance_weight,
            #             similarity=similarity,
            #             epochs=epochs,
            #             anneal_dirichlet=anneal_dirichlet,
            #             ensemble_dropout=ensemble_dropout,
            #             test_samples=test_samples,
            #             device=device)
            #
            #     # criterion=criterion,
            #     # past_models=past_models,
            #     # distance_regularization=distance_regularization,
            #     # distance_weight=distance_weight,
            #     # cosine_distance=cosine_distance,
            #     # mode_connectivity=mode_connectivity,
            #     # connect_to_last=connect_to_last)
            #

            if save:
                torch.save(backbone.state_dict(), os.path.join(experiment_path,
                                                               'bb.pt'))
                torch.save(classifiers.state_dict(), os.path.join(experiment_path,
                                                                  'classifiers.pt'))

        results = {}

        train_scores = standard_eval(model=backbone,
                                     dataset_loader=trainloader,
                                     classifier=classifiers[-1])

        test_scores = standard_eval(model=backbone,
                                    dataset_loader=testloader,
                                    classifier=classifiers[-1])

        results['train_score'] = train_scores
        results['test_score'] = test_scores

        log.info(
            'Last layer train and test scores : {}, {}'.format(train_scores,
                                                               test_scores))

        if method_name != 'standard':
            scores = branches_eval(model=backbone,
                                   dataset_loader=testloader,
                                   predictors=classifiers)
            scores = dict(scores)

            results['branch_scores'] = scores

            log.info('Branches scores: {}'.format(scores))

        if 'bernulli' in method_name:

            cumulative_threshold_scores = {}

            for epsilon in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                a, b = binary_eval(model=backbone,
                                   dataset_loader=testloader,
                                   predictors=classifiers,
                                   epsilon=epsilon,
                                   cumulative_threshold=True)

                a, b = dict(a), dict(b)

                log.info('Epsilon {} scores: {}, {}'.format(epsilon,
                                                            dict(a), dict(b)))

                cumulative_threshold_scores[epsilon] = {'scores': a,
                                                        'counters': b}

            results['cumulative_results'] = cumulative_threshold_scores

            binary_threshold_scores = {}
            for epsilon in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                a, b = binary_eval(model=backbone,
                                   dataset_loader=testloader,
                                   predictors=classifiers,
                                   epsilon=
                                           [epsilon] *
                                           (
                                                   backbone.n_branches()))

                a, b = dict(a), dict(b)

                log.info('Threshold {} scores: {}, {}'.format(epsilon, a, b))

                binary_threshold_scores[epsilon] = {'scores': a,
                                                    'counters': b}

            results['binary_results'] = binary_threshold_scores

        if method_name in ['bernulli', 'joint']:
            entropy_threshold_scores = {}

            for entropy_threshold in [0.0001, 0.01, 0.1, .2, .4, .5, .6, .7,
                                      .8]:
                a, b = entropy_eval(model=backbone,
                                    dataset_loader=testloader,
                                    predictors=classifiers,
                                    threshold=entropy_threshold)
                a, b = dict(a), dict(b)

                log.info(
                    'Entropy Threshold {} scores: {}, {}'.format(
                        entropy_threshold, a, b))

                entropy_threshold_scores[entropy_threshold] = {'scores': a,
                                                               'counters': b}

            results['entropy_results'] = entropy_threshold_scores

        with open(os.path.join(experiment_path, 'results.json'), 'w') \
                as json_file:
            json.dump(results, json_file, indent=4)

        log.info('#' * 100)


if __name__ == "__main__":
    my_app()
