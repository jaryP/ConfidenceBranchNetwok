import logging
import os
import warnings
from copy import deepcopy
from itertools import chain

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import optim

# from evaluators import accuracy_score, get_probabilities, ece_score, \
#     one_pixel_attack
# from loss_landscape_branch.evaluators import dirichlet_evaluator, \
#     exit_evaluator, dirichlet_forward
# # from loss_landscape.plots.embedding_plots import flat_embedding_plots, \
# #     flat_embedding_cosine_similarity_plot
# from loss_landscape_branch.plots.embedding_plots import flat_embedding_plots, \
#     flat_logits_plots
# from loss_landscape_branch.trainers import dirichlet_model_train, \
#     dirichlet_logits_model_train, joint_train
# from loss_landscape_branch.models.alexnet import AlexNet
# from loss_landscape_branch.utils import get_intermediate_classifiers, \
#     DirichletLogits, BranchExit, DirichletEmbeddings
from base.evaluators import binary_eval, branches_entropy, standard_eval
from base.trainer import binary_bernulli_trainer, joint_trainer, \
    standard_trainer
from utils import get_dataset, get_optimizer, get_model


# def get_model(name, image_size, classes, equalize_embedding=True):
#     name = name.lower()
#     if name == 'alexnet':
#         model = AlexNet(image_size[0])
#         # return AlexNet(input_channels), AlexNetClassifier(classes)
#     # elif 'resnet' in name:
#     #     if name == 'resnet20':
#     #         model
#     #         return resnet20(None), ResnetClassifier(classes)
#     #     else:
#     #         assert False
#     else:
#         assert False
#
#     classifiers = get_intermediate_classifiers(model,
#                                                image_size,
#                                                classes,
#                                                equalize_embedding=equalize_embedding)
#
#     return model, classifiers


def bernulli(cfg: DictConfig, logits_training: bool = False):
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

    for i in range(experiments):
        log.info('Experiment #{}'.format(i))

        torch.manual_seed(i)
        np.random.seed(i)

        experiment_path = os.path.join(path, 'exp_{}'.format(i))
        os.makedirs(experiment_path, exist_ok=True)

        train_set, test_set, input_size, classes = \
            get_dataset(name=dataset_name,
                        model_name=None,
                        augmentation=augmented_dataset)

        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        backbone, classifiers = get_model(model_name,
                                          image_size=input_size,
                                          classes=classes,
                                          get_binaries=get_binaries)

        backbone.to(device)
        classifiers.to(device)

        if os.path.exists(os.path.join(experiment_path, 'bb.pt')):
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

                pre_trained_model_path = os.path.join(pre_trained_path,
                                                      '{}.pt'.format(i))

                if os.path.exists(pre_trained_model_path):
                    log.info('Pre trained model loaded')
                    log.info(pre_trained_model_path)

                    backbone.load_state_dict(
                        torch.load(pre_trained_model_path,
                                   map_location=device))
                else:

                    os.makedirs(pre_trained_path, exist_ok=True)

                    log.info('Training the base model')

                    _classifiers = deepcopy(classifiers)
                    parameters = chain(backbone.parameters(),
                                       _classifiers.parameters())

                    optimizer = get_optimizer(parameters=parameters,
                                              name='adam',
                                              lr=0.001,
                                              momentum=None,
                                              weight_decay=0)

                    res = standard_trainer(model=backbone,
                                           predictors=_classifiers,
                                           optimizer=optimizer,
                                           train_loader=trainloader,
                                           epochs=epochs,
                                           scheduler=None,
                                           early_stopping=None,
                                           test_loader=testloader,
                                           eval_loader=None, device=device)[0]

                    backbone_dict, classifiers_dict = res

                    backbone.load_state_dict(backbone_dict)
                    # classifiers.load_state_dict(classifiers_dict)

                    torch.save(backbone.state_dict(), pre_trained_model_path)

                    log.info('Pre trained model Saved.')

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

                res = binary_bernulli_trainer(model=backbone,
                                              predictors=classifiers,
                                              bernulli_models=classifiers,
                                              optimizer=optimizer,
                                              train_loader=trainloader,
                                              epochs=epochs,
                                              prior_parameters=priors,
                                              joint_type=joint_type,
                                              # cost_reg=1e-3,
                                              # use_mmd=False,
                                              # scheduler=None,
                                              # early_stopping=None,
                                              test_loader=testloader,
                                              fixed_bernulli=fixed_bernulli,
                                              # eval_loader=,
                                              device=device)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'joint':
                joint_type = method_cfg.get('joint_type', 'predictions')
                weights = method_cfg.get('weights', None)
                train_weights = method_cfg.get('train_weights', False)

                if train_weights:
                    weights = torch.tensor(weights, device=device,
                                           dtype=torch.float)

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
                                    epochs=epochs, train_weights=train_weights,
                                    scheduler=None, joint_type=joint_type,
                                    early_stopping=None, test_loader=testloader,
                                    eval_loader=None, device=device)[0]

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
                                       eval_loader=None, device=device)[0]

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

            torch.save(backbone.state_dict(), os.path.join(experiment_path,
                                                           'bb.pt'))
            torch.save(classifiers.state_dict(), os.path.join(experiment_path,
                                                              'classifiers.pt'))

        train_scores = standard_eval(model=backbone,
                                     dataset_loader=trainloader,
                                     classifier=classifiers[-1],
                                     device=device)

        test_scores = standard_eval(model=backbone,
                                    dataset_loader=testloader,
                                    classifier=classifiers[-1],
                                    device=device)

        log.info(
            'Last layer train and test scores : {}, {}'.format(train_scores,
                                                               test_scores))

        if method_name == 'bernulli':

            for epsilon in [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]:
                a, b = binary_eval(model=backbone,
                                   dataset_loader=testloader,
                                   predictors=classifiers,
                                   device=device,
                                   epsilon=epsilon,
                                   cumulative_threshold=True)

                log.info('Epsilon {} scores: {}, {}'.format(epsilon,
                                                            dict(a), dict(b)))

            for epsilon in [.2, .4, .5, .6, .7, .8]:
                branches_score, counter = binary_eval(model=backbone,
                                                      dataset_loader=testloader,
                                                      predictors=classifiers,
                                                      device=device,
                                                      epsilon=[0.6] +
                                                              [epsilon] *
                                                              (
                                                                      backbone.n_branches() - 1))

                log.info('Threshold {} scores: {}, {}'.format(epsilon,
                                                              dict(
                                                                  branches_score),
                                                              dict(counter)))

        if method_name in ['bernulli', 'joint']:
            for entropy_threshold in [0.0001, 0.01, 0.1, .2, .4, .5, .6, .7,
                                      .8]:
                branches_score, counter = branches_entropy(model=backbone,
                                                           dataset_loader=testloader,
                                                           predictors=classifiers,
                                                           device=device,
                                                           threshold=entropy_threshold)

                log.info(
                    'Entropy Threshold {} scores: {}, {}'.format(
                        entropy_threshold,
                        dict(
                            branches_score),
                        dict(counter)))

        # log.info('Evaluation process')
        #
        # ff = BranchExit(model=backbone,
        #                 classifiers=classifiers,
        #                 exit_to_use=-1,
        #                 use_last_classifier=not logits_training)
        #
        # for i in range(backbone.n_branches()):
        #     ff.exit_to_use = i
        #     score = accuracy_score(testloader, ff, device)[0]
        #
        #     # score, _, _ = exit_evaluator(backbone=backbone,
        #     #                              classifiers=classifiers,
        #     #                              dataloader=testloader,
        #     #                              exit_to_use=i,
        #     #                              use_last_classifier=not logits_training,
        #     #                              device=device)
        #
        #     log.info('Score obtained using exit #{}: {}'.format(i + 1,
        #                                                         score))
        #
        # # score, total, correct = dirichlet_evaluator(backbone=backbone,
        # #                                             # backbone2=backbone2,
        # #                                             # backbone3=backbone3,
        # #                                             classifiers=classifiers,
        # #                                             dataloader=testloader,
        # #                                             test_samples=test_samples,
        # #                                             combine_logits=logits_training,
        # #                                             device=device)
        #
        # if logits_training:
        #     ff = DirichletLogits(model=backbone,
        #                          classifiers=classifiers,
        #                          samples=test_samples)
        # else:
        #     ff = DirichletEmbeddings(model=backbone,
        #                              classifiers=classifiers,
        #                              samples=test_samples)
        #
        # one_pixel_attack(testloader, ff, device)
        #
        # score = accuracy_score(testloader, ff, device)[0]
        #
        # log.info('Dirichlet score: {}'.format(score))
        #
        # # ground_truth, predicted_labels, probs = dirichlet_forward(
        # #     backbone=backbone,
        # #     # backbone2=backbone2,
        # #     # backbone3=backbone3,
        # #     classifiers=classifiers,
        # #     dataloader=testloader,
        # #     test_samples=test_samples,
        # #     combine_logits=logits_training,
        # #     device=device)
        #
        # # ground_truth, predicted_labels, probs = \
        # #     get_probabilities(testloader, ff, device)
        #
        # ece, _, _, _ = ece_score(testloader, ff, device)
        # log.info('ECE score: {}'.format(ece))
        #
        # # ff = DirichletLogits(model=backbone,
        # #                      classifiers=classifiers,
        # #                      samples=test_samples)
        # # score = accuracy_score(testloader, ff, device)[0]
        # # input(score)
        # # score, total, correct = dirichlet_evaluator(backbone1=backbone1,
        # #                                             backbone2=backbone2,
        # #                                             backbone3=backbone3,
        # #                                             classifier=classifier,
        # #                                             dataloader=testloader,
        # #                                             device=device)
        # #
        # # log.info('Dirichlet score: {}'.format(score))
        # #
        # # if plot:
        # #     log.info('Plotting.')
        # #     images_path = os.path.join(experiment_path, 'images')
        # #     os.makedirs(images_path, exist_ok=True)
        # #
        # #     if not os.path.exists(
        # #             os.path.join(images_path, "flatten_embs.pdf")):
        # #         loss_figure, score_figure = flat_embedding_plots(backbone1,
        # #                                                          backbone2,
        # #                                                          backbone3,
        # #                                                          classifier,
        # #                                                          testloader,
        # #                                                          device=device)
        # #
        # #         loss_figure.savefig(
        # #             os.path.join(images_path, "flatten_embs.pdf"),
        # #             bbox_inches='tight')
        # #         plt.close(loss_figure)
        # #
        # #         score_figure.savefig(
        # #             os.path.join(images_path, "flatten_embs_score.pdf"),
        # #             bbox_inches='tight')
        # #         plt.close(score_figure)
        # #
        # #     # if not os.path.exists(os.path.join(images_path, "weights_plot.pdf")):
        # #     #     figure, figure_score = weights_plot(b1, b2, b3, classifier,
        # #     #                                         testloader,
        # #     #                                         criterion=criterion,
        # #     #                                         device=device)
        # #     #
        # #     #     figure.savefig(os.path.join(images_path, "weights_plot.pdf"),
        # #     #                    bbox_inches='tight')
        # #     #     plt.close(figure)
        # #     #
        # #     #     figure_score.savefig(os.path.join(images_path, "weights_score.pdf"),
        # #     #                          bbox_inches='tight')
        # #     #     plt.close(figure_score)
        # #     #
        # #     # if not os.path.exists(
        # #     #         os.path.join(images_path, "model_similarity.pdf")):
        # #     #     figure = model_cosine_similarity_plot(b1, b2, b3, classifier,
        # #     #                                           testloader,
        # #     #                                           criterion=criterion,
        # #     #                                           device=device)
        # #     #
        # #     #     figure.savefig(os.path.join(images_path, "model_similarity.pdf"),
        # #     #                    bbox_inches='tight')
        # #     #     plt.close(figure)
        # #
        # #     if not os.path.exists(
        # #             os.path.join(images_path, "flatten_embs_similarity.pdf")):
        # #         figure = flat_embedding_cosine_similarity_plot(backbone1,
        # #                                                        backbone2,
        # #                                                        backbone3,
        # #                                                        classifier,
        # #                                                        testloader,
        # #                                                        device=device)
        # #         figure.savefig(
        # #             os.path.join(images_path, "flatten_embs_similarity.pdf"),
        # #             bbox_inches='tight')
        # #         plt.close(figure)
        # #
        # #     # if not os.path.exists(
        # #     #         os.path.join(images_path, "logits_similarity.pdf")):
        # #     #     figure = logits_cosine_similarity_plot(b1, b2, b3, classifier,
        # #     #                                            testloader,
        # #     #                                            criterion=criterion,
        # #     #                                            device=device)
        # #     #     figure.savefig(
        # #     #         os.path.join(images_path, "logits_similarity.pdf"),
        # #     #         bbox_inches='tight')
        # #     #     plt.close(figure)
        # #     #
        # #     # if not os.path.exists(
        # #     #         os.path.join(images_path, "similarity_0.pdf")):
        # #     #
        # #     #     figures = embedding_cosine_similarity_plot(b1, b2, b3,
        # #     #                                                classifier,
        # #     #                                                testloader,
        # #     #                                                criterion=criterion,
        # #     #                                                device=device)
        # #     #
        # #     #     for label, figure in figures:
        # #     #         figure.savefig(
        # #     #             os.path.join(images_path,
        # #     #                          "similarity_{}.pdf".format(label)),
        # #     #             bbox_inches='tight')
        # #     #         plt.close(figure)
        # #     #
        # #     # if not os.path.exists(
        # #     #         os.path.join(images_path, "loss_space_class_0.pdf")):
        # #     #
        # #     #     figures = classes_embedding_plots(b1, b2, b3, classifier,
        # #     #                                       testloader,
        # #     #                                       criterion=criterion,
        # #     #                                       device=device)
        # #     #
        # #     #     for label, figure in figures:
        # #     #         figure.savefig(
        # #     #             os.path.join(images_path,
        # #     #                          "loss_space_class{}.pdf".format(label)),
        # #     #             bbox_inches='tight')
        # #     #         plt.close(figure)
        #
        # if plot:
        #     log.info('Plotting.')
        #     images_path = os.path.join(experiment_path, 'images')
        #     os.makedirs(images_path, exist_ok=True)
        #
        #     if not os.path.exists(
        #             os.path.join(images_path, "flatten_embs.pdf")):
        #         loss_figure, score_figure = flat_embedding_plots(backbone,
        #                                                          classifiers,
        #                                                          testloader,
        #                                                          device=device)
        #
        #         loss_figure.savefig(
        #             os.path.join(images_path, "flatten_embs.pdf"),
        #             bbox_inches='tight')
        #         plt.close(loss_figure)
        #
        #         score_figure.savefig(
        #             os.path.join(images_path, "flatten_embs_score.pdf"),
        #             bbox_inches='tight')
        #         plt.close(score_figure)
        #
        #     if logits_training:
        #         loss_figure, score_figure = flat_logits_plots(backbone,
        #                                                       classifiers,
        #                                                       testloader,
        #                                                       grid_points=100,
        #                                                       plot_offsets=1,
        #                                                       device=device)
        #
        #         loss_figure.savefig(
        #             os.path.join(images_path, "flatten_logits.pdf"),
        #             bbox_inches='tight')
        #         plt.close(loss_figure)
        #
        #         score_figure.savefig(
        #             os.path.join(images_path, "flatten_logits_score.pdf"),
        #             bbox_inches='tight')
        #         plt.close(score_figure)
        #
        #     # if not os.path.exists(os.path.join(images_path, "weights_plot.pdf")):
        #     #     figure, figure_score = weights_plot(b1, b2, b3, classifier,
        #     #                                         testloader,
        #     #                                         criterion=criterion,
        #     #                                         device=device)
        #     #
        #     #     figure.savefig(os.path.join(images_path, "weights_plot.pdf"),
        #     #                    bbox_inches='tight')
        #     #     plt.close(figure)
        #     #
        #     #     figure_score.savefig(os.path.join(images_path, "weights_score.pdf"),
        #     #                          bbox_inches='tight')
        #     #     plt.close(figure_score)
        #     #
        #     # if not os.path.exists(
        #     #         os.path.join(images_path, "model_similarity.pdf")):
        #     #     figure = model_cosine_similarity_plot(b1, b2, b3, classifier,
        #     #                                           testloader,
        #     #                                           criterion=criterion,
        #     #                                           device=device)
        #     #
        #     #     figure.savefig(os.path.join(images_path, "model_similarity.pdf"),
        #     #                    bbox_inches='tight')
        #     #     plt.close(figure)
        #
        #     # if not os.path.exists(
        #     #         os.path.join(images_path, "flatten_embs_similarity.pdf")):
        #     #     figure = flat_embedding_cosine_similarity_plot(backbone1,
        #     #                                                    backbone2,
        #     #                                                    backbone3,
        #     #                                                    classifier,
        #     #                                                    testloader,
        #     #                                                    device=device)
        #     #     figure.savefig(
        #     #         os.path.join(images_path, "flatten_embs_similarity.pdf"),
        #     #         bbox_inches='tight')
        #     #     plt.close(figure)

        log.info('#' * 100)


# def joint(cfg: DictConfig):
#     log = logging.getLogger(__name__)
#     log.info(OmegaConf.to_yaml(cfg))
#
#     model_cfg = cfg['model']
#     model_name = model_cfg['name']
#
#     dataset_cfg = cfg['dataset']
#     dataset_name = dataset_cfg['name']
#     augmented_dataset = dataset_cfg.get('augment', False)
#
#     experiment_cfg = cfg['experiment']
#     load, save, path, experiments = experiment_cfg.get('load', True), \
#                                     experiment_cfg.get('save', True), \
#                                     experiment_cfg.get('path', None), \
#                                     experiment_cfg.get('experiments', 1)
#
#     plot = experiment_cfg.get('plot', False)
#
#     method_cfg = cfg['method']
#     weights = method_cfg.get('weights', None)
#
#     training_cfg = cfg['training']
#     epochs, batch_size, device = training_cfg['epochs'], \
#                                  training_cfg['batch_size'], \
#                                  training_cfg.get('device', 'cpu')
#
#     optimizer_cfg = cfg['optimizer']
#     optimizer, lr, momentum, weight_decay = optimizer_cfg.get('optimizer',
#                                                               'sgd'), \
#                                             optimizer_cfg.get('lr', 1e-1), \
#                                             optimizer_cfg.get('momentum', 0.9), \
#                                             optimizer_cfg.get('weight_decay', 0)
#
#     if torch.cuda.is_available() and device != 'cpu':
#         torch.cuda.set_device(device)
#         device = 'cuda:{}'.format(device)
#     else:
#         warnings.warn("Device not found or CUDA not available.")
#
#     device = torch.device(device)
#
#     if not isinstance(experiments, int):
#         raise ValueError('experiments argument must be integer: {} given.'
#                          .format(experiments))
#
#     if path is None:
#         path = os.getcwd()
#     else:
#         os.chdir(path)
#         os.makedirs(path, exist_ok=True)
#
#     for i in range(experiments):
#         log.info('Experiment #{}'.format(i))
#
#         torch.manual_seed(i)
#         np.random.seed(i)
#
#         experiment_path = os.path.join(path, 'exp_{}'.format(i))
#         os.makedirs(experiment_path, exist_ok=True)
#
#         train_set, test_set, input_size, classes = \
#             get_dataset(name=dataset_name,
#                         model_name=None,
#                         augmentation=augmented_dataset)
#
#         trainloader = torch.utils.data.DataLoader(train_set,
#                                                   batch_size=batch_size,
#                                                   shuffle=True)
#
#         testloader = torch.utils.data.DataLoader(test_set,
#                                                  batch_size=batch_size,
#                                                  shuffle=False)
#
#         backbone, classifiers = get_model(model_name,
#                                           image_size=input_size,
#                                           classes=classes)
#
#         if os.path.exists(os.path.join(experiment_path, 'bb.pt')):
#             backbone.load_state_dict(torch.load(
#                 os.path.join(experiment_path, 'bb.pt')))
#             classifiers.load_state_dict(torch.load(
#                 os.path.join(experiment_path, 'classifiers.pt')))
#         else:
#             log.info('Training started.')
#
#             # optimizer = optim.SGD(chain(backbone1.parameters(),
#             #                             backbone2.parameters(),
#             #                             backbone3.parameters(),
#             #                             classifier.parameters()), lr=0.01,
#             #                       momentum=0.9)
#
#             parameters = chain(backbone.parameters(),
#                                classifiers.parameters())
#
#             optimizer = get_optimizer(parameters=parameters, name=optimizer,
#                                       lr=lr,
#                                       momentum=momentum,
#                                       weight_decay=weight_decay)
#
#             backbone, classifiers = joint_train(
#                 backbone=backbone,
#                 classifiers=classifiers,
#                 trainloader=trainloader,
#                 testloader=testloader,
#                 weights=weights,
#                 optimizer=optimizer,
#                 epochs=epochs,
#                 device=device)
#
#             # criterion=criterion,
#             # past_models=past_models,
#             # distance_regularization=distance_regularization,
#             # distance_weight=distance_weight,
#             # cosine_distance=cosine_distance,
#             # mode_connectivity=mode_connectivity,
#             # connect_to_last=connect_to_last)
#
#             torch.save(backbone.state_dict(), os.path.join(experiment_path,
#                                                            'bb.pt'))
#             torch.save(classifiers.state_dict(), os.path.join(experiment_path,
#                                                               'classifiers.pt'))
#         log.info('Evaluation process')
#
#         for i in range(backbone.n_branches()):
#             score, _, _ = exit_evaluator(backbone=backbone,
#                                          classifiers=classifiers,
#                                          dataloader=testloader,
#                                          exit_to_use=i,
#                                          use_last_classifier=False,
#                                          device=device)
#
#             log.info('Score obtained using exit #{}: {}'.format(i + 1,
#                                                                 score))
#
#         # score, total, correct = dirichlet_evaluator(backbone=backbone,
#         #                                             # backbone2=backbone2,
#         #                                             # backbone3=backbone3,
#         #                                             classifiers=classifiers,
#         #                                             dataloader=testloader,
#         #                                             test_samples=test_samples,
#         #                                             combine_logits=logits_training,
#         #                                             device=device)
#         # log.info('Dirichlet score: {}'.format(score))
#
#         # score, total, correct = dirichlet_evaluator(backbone1=backbone1,
#         #                                             backbone2=backbone2,
#         #                                             backbone3=backbone3,
#         #                                             classifier=classifier,
#         #                                             dataloader=testloader,
#         #                                             device=device)
#         #
#         # log.info('Dirichlet score: {}'.format(score))
#         #
#         # if plot:
#         #     log.info('Plotting.')
#         #     images_path = os.path.join(experiment_path, 'images')
#         #     os.makedirs(images_path, exist_ok=True)
#         #
#         #     if not os.path.exists(
#         #             os.path.join(images_path, "flatten_embs.pdf")):
#         #         loss_figure, score_figure = flat_embedding_plots(backbone1,
#         #                                                          backbone2,
#         #                                                          backbone3,
#         #                                                          classifier,
#         #                                                          testloader,
#         #                                                          device=device)
#         #
#         #         loss_figure.savefig(
#         #             os.path.join(images_path, "flatten_embs.pdf"),
#         #             bbox_inches='tight')
#         #         plt.close(loss_figure)
#         #
#         #         score_figure.savefig(
#         #             os.path.join(images_path, "flatten_embs_score.pdf"),
#         #             bbox_inches='tight')
#         #         plt.close(score_figure)
#         #
#         #     # if not os.path.exists(os.path.join(images_path, "weights_plot.pdf")):
#         #     #     figure, figure_score = weights_plot(b1, b2, b3, classifier,
#         #     #                                         testloader,
#         #     #                                         criterion=criterion,
#         #     #                                         device=device)
#         #     #
#         #     #     figure.savefig(os.path.join(images_path, "weights_plot.pdf"),
#         #     #                    bbox_inches='tight')
#         #     #     plt.close(figure)
#         #     #
#         #     #     figure_score.savefig(os.path.join(images_path, "weights_score.pdf"),
#         #     #                          bbox_inches='tight')
#         #     #     plt.close(figure_score)
#         #     #
#         #     # if not os.path.exists(
#         #     #         os.path.join(images_path, "model_similarity.pdf")):
#         #     #     figure = model_cosine_similarity_plot(b1, b2, b3, classifier,
#         #     #                                           testloader,
#         #     #                                           criterion=criterion,
#         #     #                                           device=device)
#         #     #
#         #     #     figure.savefig(os.path.join(images_path, "model_similarity.pdf"),
#         #     #                    bbox_inches='tight')
#         #     #     plt.close(figure)
#         #
#         #     if not os.path.exists(
#         #             os.path.join(images_path, "flatten_embs_similarity.pdf")):
#         #         figure = flat_embedding_cosine_similarity_plot(backbone1,
#         #                                                        backbone2,
#         #                                                        backbone3,
#         #                                                        classifier,
#         #                                                        testloader,
#         #                                                        device=device)
#         #         figure.savefig(
#         #             os.path.join(images_path, "flatten_embs_similarity.pdf"),
#         #             bbox_inches='tight')
#         #         plt.close(figure)
#         #
#         #     # if not os.path.exists(
#         #     #         os.path.join(images_path, "logits_similarity.pdf")):
#         #     #     figure = logits_cosine_similarity_plot(b1, b2, b3, classifier,
#         #     #                                            testloader,
#         #     #                                            criterion=criterion,
#         #     #                                            device=device)
#         #     #     figure.savefig(
#         #     #         os.path.join(images_path, "logits_similarity.pdf"),
#         #     #         bbox_inches='tight')
#         #     #     plt.close(figure)
#         #     #
#         #     # if not os.path.exists(
#         #     #         os.path.join(images_path, "similarity_0.pdf")):
#         #     #
#         #     #     figures = embedding_cosine_similarity_plot(b1, b2, b3,
#         #     #                                                classifier,
#         #     #                                                testloader,
#         #     #                                                criterion=criterion,
#         #     #                                                device=device)
#         #     #
#         #     #     for label, figure in figures:
#         #     #         figure.savefig(
#         #     #             os.path.join(images_path,
#         #     #                          "similarity_{}.pdf".format(label)),
#         #     #             bbox_inches='tight')
#         #     #         plt.close(figure)
#         #     #
#         #     # if not os.path.exists(
#         #     #         os.path.join(images_path, "loss_space_class_0.pdf")):
#         #     #
#         #     #     figures = classes_embedding_plots(b1, b2, b3, classifier,
#         #     #                                       testloader,
#         #     #                                       criterion=criterion,
#         #     #                                       device=device)
#         #     #
#         #     #     for label, figure in figures:
#         #     #         figure.savefig(
#         #     #             os.path.join(images_path,
#         #     #                          "loss_space_class{}.pdf".format(label)),
#         #     #             bbox_inches='tight')
#         #     #         plt.close(figure)
#
#         if plot:
#             log.info('Plotting.')
#             images_path = os.path.join(experiment_path, 'images')
#             os.makedirs(images_path, exist_ok=True)
#             #
#             # if not os.path.exists(
#             #         os.path.join(images_path, "flatten_embs.pdf")):
#             #     loss_figure, score_figure = flat_embedding_plots(backbone,
#             #                                                      classifiers,
#             #                                                      testloader,
#             #                                                      device=device)
#             #
#             #     loss_figure.savefig(
#             #         os.path.join(images_path, "flatten_embs.pdf"),
#             #         bbox_inches='tight')
#             #     plt.close(loss_figure)
#             #
#             #     score_figure.savefig(
#             #         os.path.join(images_path, "flatten_embs_score.pdf"),
#             #         bbox_inches='tight')
#             #     plt.close(score_figure)
#
#             # if logits_training:
#             loss_figure, score_figure = flat_logits_plots(backbone,
#                                                           classifiers,
#                                                           testloader,
#                                                           grid_points=100,
#                                                           plot_offsets=1,
#                                                           device=device)
#
#             loss_figure.savefig(
#                 os.path.join(images_path, "flatten_logits.pdf"),
#                 bbox_inches='tight')
#             plt.close(loss_figure)
#
#             score_figure.savefig(
#                 os.path.join(images_path, "flatten_logits_score.pdf"),
#                 bbox_inches='tight')
#             plt.close(score_figure)
#
#             # if not os.path.exists(os.path.join(images_path, "weights_plot.pdf")):
#             #     figure, figure_score = weights_plot(b1, b2, b3, classifier,
#             #                                         testloader,
#             #                                         criterion=criterion,
#             #                                         device=device)
#             #
#             #     figure.savefig(os.path.join(images_path, "weights_plot.pdf"),
#             #                    bbox_inches='tight')
#             #     plt.close(figure)
#             #
#             #     figure_score.savefig(os.path.join(images_path, "weights_score.pdf"),
#             #                          bbox_inches='tight')
#             #     plt.close(figure_score)
#             #
#             # if not os.path.exists(
#             #         os.path.join(images_path, "model_similarity.pdf")):
#             #     figure = model_cosine_similarity_plot(b1, b2, b3, classifier,
#             #                                           testloader,
#             #                                           criterion=criterion,
#             #                                           device=device)
#             #
#             #     figure.savefig(os.path.join(images_path, "model_similarity.pdf"),
#             #                    bbox_inches='tight')
#             #     plt.close(figure)
#
#             # if not os.path.exists(
#             #         os.path.join(images_path, "flatten_embs_similarity.pdf")):
#             #     figure = flat_embedding_cosine_similarity_plot(backbone1,
#             #                                                    backbone2,
#             #                                                    backbone3,
#             #                                                    classifier,
#             #                                                    testloader,
#             #                                                    device=device)
#             #     figure.savefig(
#             #         os.path.join(images_path, "flatten_embs_similarity.pdf"),
#             #         bbox_inches='tight')
#             #     plt.close(figure)
#
#         log.info('#' * 100)


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    # method_name = cfg['method']['name']
    # if method_name == 'bernulli':
    bernulli(cfg)
    # elif method_name == 'dirichlet_logits':
    # dirichlet(cfg, logits_training=True)
    # elif method_name == 'joint':
    #     joint(cfg)
    # else:
    #     raise ValueError('Supported methods are: [dirichlet, dirichlet_logits]')


if __name__ == "__main__":
    my_app()
