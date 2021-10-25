import json
import os
import re

import hydra
import numpy as np
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import cm


def plot_scores(results: dict, branches=5):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_ylim(-0.1, 1.05)
    ax2.set_ylim(-0.1, 1.05)

    ax1.set_ylabel('Branches scores')
    ax2.set_ylabel('Branches counters')
    ax2.set_xlabel('Threshold')

    keys = sorted(map(float, results.keys()))
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    legend = set()

    for i, k in enumerate(keys):
        k = str(k)
        r = results[k]

        scores = r['scores']
        counters = r['counters']
        tot = sum(counters.values())

        x = i - 0.4 + (0.8 / 6) * 6
        ax1.bar(x, scores['global'], width=0.8 / 6,
                color=colors[-1], label='Acc' if i == 0 else '')

        for j in range(5):
            kj = str(j)

            if kj in scores:
                x = i - 0.4 + (0.8 / 6) * j
                ax1.bar(x, scores[kj], width=0.8 / 6,
                        color=colors[j],
                        label='B {}'.format(j) if j not in legend else '')
                legend.add(j)

            if kj in counters:
                x = i - 0.4 + (0.8 / 5) * j
                ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
                        color=colors[
                            j],
                        label='B {}'.format(j) if j not in legend else '')
                legend.add(j)

    for i in np.arange(0, 1.1, 0.1):
        ax2.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
        ax1.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)

    ax2.set_xticklabels([''] + keys)
    fig.legend()

    return fig


def plot_heatmap(results: dict, branches=5):
    keys = sorted(map(float, results.keys()))
    # colors = ['b', 'g', 'r', 'c', 'm', 'k']
    # legend = set()

    matrix_scores = np.zeros((len(keys), 5 + 1))
    matrix_counters = np.zeros((len(keys), 5))

    for i, k in enumerate(keys):
        k = str(k)
        r = results[k]

        scores = r['scores']
        counters = r['counters']
        tot = sum(counters.values())

        for j in range(5):
            kj = str(j)

            matrix_scores[i][j] = scores.get(kj, 0)
            matrix_counters[i][j] = counters.get(kj, 0)

        matrix_scores[i][-1] = scores['global']

    tot = matrix_counters.sum(1)[0]
    # print(tot)
    matrix_counters /= tot

    #     x = i - 0.4 + (0.8 / 6) * 6
    #     ax1.bar(x, scores['global'], width=0.8 / 6,
    #             color=colors[-1], label='Acc' if i == 0 else '')
    #
    #     for j in range(5):
    #         kj = str(j)
    #
    #         if kj in scores:
    #             x = i - 0.4 + (0.8 / 6) * j
    #             ax1.bar(x, scores[kj], width=0.8 / 6,
    #                     color=colors[j],
    #                     label='B {}'.format(j) if j not in legend else '')
    #             legend.add(j)
    #
    #         if kj in counters:
    #             x = i - 0.4 + (0.8 / 5) * j
    #             ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
    #                     color=colors[
    #                         j], label='B {}'.format(j) if j not in legend else '')
    #             legend.add(j)
    #
    # for i in np.arange(0, 1.1, 0.1):
    #     ax2.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
    #     ax1.axhline(y=i, color='k', linestyle='--', alpha=0.1, linewidth=1)
    #
    # ax2.set_xticklabels([''] + keys)
    # fig.legend()

    # return fig

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    matrix_scores = np.around(matrix_scores * 100, 1)
    matrix_counters = np.around(matrix_counters * 100, 1)

    ax1.imshow(matrix_scores, cmap=cm.inferno, vmin=0, vmax=100)
    ax2.imshow(matrix_counters, cmap=cm.Blues, vmin=0, vmax=100)

    # ax1.set_ylim(-0.1, 1.05)
    # ax2.set_ylim(-0.1, 1.05)

    ax1.set_xlabel('Branches scores')
    ax1.set_ylabel('Threshold')
    ax1.set_yticklabels([''] + keys)
    ax1.set_xticklabels([''] + list(range(5)) + ['Global'])

    ax2.set_xlabel('Branches counters')
    ax2.set_ylabel('Threshold')
    ax2.set_yticklabels([''] + keys)

    for i in range(len(keys)):
        for j in range(matrix_scores.shape[1]):
            text = ax1.text(j, i, matrix_scores[i, j],
                            ha="center", va="center", color="w" if 0 > matrix_scores[i, j] > 50 else 'k')

    for i in range(len(keys)):
        for j in range(matrix_counters.shape[1]):
            text = ax2.text(j, i, matrix_counters[i, j],
                            ha="center", va="center", color="w" if 0 == matrix_counters[i, j] or matrix_counters[i, j] > 50 else 'k')

    return fig1, fig2


def my_app() -> None:
    paths = sys.argv[1:]

    # # log = logging.getLogger(__name__)
    #
    # # model_cfg = cfg['model']
    # # # model_name = model_cfg['name']
    # #
    # # dataset_cfg = cfg['dataset']
    # # dataset_name = dataset_cfg['name']
    # # augmented_dataset = dataset_cfg.get('augment', False)
    #
    # experiment_cfg = cfg['experiment']
    # load, save, path, experiments = experiment_cfg.get('load', True), \
    #                                 experiment_cfg.get('save', True), \
    #                                 experiment_cfg.get('path', None), \
    #                                 experiment_cfg.get('experiments', 1)
    #
    # # plot = experiment_cfg.get('plot', False)
    # #
    # # method_cfg = cfg['method']
    # # method_name = method_cfg['name']
    # #
    # # get_binaries = True if method_name == 'bernulli' else False
    # #
    # # # trainer = None
    # #
    # # # if method_name == 'bernulli':
    # # #     trainer = binary_bernulli_trainer
    # #
    # # # distance_regularization, distance_weight, similarity = method_cfg.get(
    # # #     'distance_regularization', True), \
    # # #                                                        method_cfg.get(
    # # #                                                            'distance_weight',
    # # #                                                            1), \
    # # #                                                        method_cfg[
    # # #                                                            'similarity']
    # # # ensemble_dropout = method_cfg.get('ensemble_dropout', 0)
    # # # anneal_dirichlet = method_cfg.get('anneal_dirichlet', True)
    # # # test_samples = method_cfg.get('test_samples', 1)
    # #
    # # training_cfg = cfg['training']
    # # epochs, batch_size, device = training_cfg['epochs'], \
    # #                              training_cfg['batch_size'], \
    # #                              training_cfg.get('device', 'cpu')
    # # eval_percentage = training_cfg.get('eval_split', None)
    # #
    # # optimizer_cfg = cfg['optimizer']
    # # optimizer_name, lr, momentum, weight_decay = optimizer_cfg.get('optimizer',
    # #                                                                'sgd'), \
    # #                                              optimizer_cfg.get('lr', 1e-1), \
    # #                                              optimizer_cfg.get('momentum',
    # #                                                                0.9), \
    # #                                              optimizer_cfg.get(
    # #                                                  'weight_decay', 0)
    # #
    # # if torch.cuda.is_available() and device != 'cpu':
    # #     torch.cuda.set_device(device)
    # #     device = 'cuda:{}'.format(device)
    # # else:
    # #     warnings.warn("Device not found or CUDA not available.")
    # #
    # # device = torch.device(device)
    #
    # if not isinstance(experiments, int):
    #     raise ValueError('experiments argument must be integer: {} given.'
    #                      .format(experiments))
    #
    # if path is None:
    #     path = os.getcwd()
    # else:
    #     os.chdir(path)
    #     os.makedirs(path, exist_ok=True)
    #

    # for experiment in range(experiments):
    # log.info('Experiment #{}'.format(experiment))

    # torch.manual_seed(experiment)
    # np.random.seed(experiment)

    for path in paths:
        for experiment_path in [f for f in os.listdir(path)
                                if re.match(r'exp_[0-9]', f)]:

            # eval_loader = None
            # train_set, test_set, input_size, classes = \
            #     get_dataset(name=dataset_name,
            #                 model_name=None,
            #                 augmentation=augmented_dataset)
            #
            # if eval_percentage is not None and eval_percentage > 0:
            #     assert eval_percentage < 1
            #     train_len = len(train_set)
            #     eval_len = int(train_len * eval_percentage)
            #     train_len = train_len - eval_len
            #
            #     train_set, eval = torch.utils.data.random_split(train_set,
            #                                                     [train_len, eval_len])
            #     # train_loader = torch.utils.data.DataLoader(dataset=train,
            #     #                                            batch_size=batch_size,
            #     #                                            shuffle=True)
            #     eval_loader = torch.utils.data.DataLoader(dataset=eval,
            #                                               batch_size=batch_size,
            #                                               shuffle=False)
            #
            #     # logger.info('Train dataset size: {}'.format(len(train)))
            #     # logger.info('Test dataset size: {}'.format(len(test)))
            #     # logger.info(
            #     #     'Eval dataset created, having size: {}'.format(len(eval)))
            #
            # trainloader = torch.utils.data.DataLoader(train_set,
            #                                           batch_size=batch_size,
            #                                           shuffle=True)
            #
            # testloader = torch.utils.data.DataLoader(test_set,
            #                                          batch_size=batch_size,
            #                                          shuffle=False)
            #
            # backbone, classifiers = get_model(model_name, image_size=input_size,
            #                                   classes=classes,
            #                                   get_binaries=get_binaries)
            full_path = os.path.join(path, experiment_path)

            if os.path.exists(os.path.join(full_path, 'results.json')):
                # log.info('Model loaded')
                with open(os.path.join(full_path, 'results.json'), 'r') \
                        as json_file:
                    results = json.load(json_file)

                plot_dirs = os.path.join(full_path, 'plots')
                os.makedirs(plot_dirs, exist_ok=True)

                if 'binary_results' in results:
                    binary_scores_fig = plot_scores(results['binary_results'])
                    binary_scores_fig.savefig(os.path.join(
                        plot_dirs, 'binary_results.pdf'))

                    f1, f2 = plot_heatmap(results['binary_results'])
                    f1.savefig(os.path.join(plot_dirs,
                                            'binary_results_hm_scores.pdf'))
                    f2.savefig(os.path.join(plot_dirs,
                                            'binary_results_hm_counters.pdf'))

                    plt.close()

                if 'cumulative_results' in results:
                    cumulative_scores_fig = plot_scores(
                        results['cumulative_results'])
                    cumulative_scores_fig.savefig(os.path.join(
                        plot_dirs, 'cumulative_results.pdf'))

                    f1, f2 = plot_heatmap(results['cumulative_results'])
                    f1.savefig(os.path.join(plot_dirs,
                                            'cumulative_results_hm_scores.pdf'))
                    f2.savefig(os.path.join(plot_dirs,
                                            'cumulative_results_hm_counters.pdf'))
                    plt.close()

                if 'entropy_results' in results:
                    cumulative_scores_fig = plot_scores(
                        results['entropy_results'])
                    cumulative_scores_fig.savefig(os.path.join(
                        plot_dirs, 'entropy_results.pdf'))

                    f1, f2 = plot_heatmap(results['entropy_results'])
                    f1.savefig(os.path.join(plot_dirs,
                                            'entropy_results_hm_scores.pdf'))
                    f2.savefig(os.path.join(plot_dirs,
                                            'entropy_results_hm_counters.pdf'))
                    plt.close()

                # plt.show()

            # binary_results = results['binary_results']
            #
            # fig, (ax1, ax2) = plt.subplots(2, sharex=True, num=0)
            # ax1.set_ylim(-0.1, 1)
            # ax2.set_ylim(-0.1, 1)
            #
            # ax1.set_ylabel('Branches scores')
            # ax2.set_ylabel('Branches counters')
            # ax2.set_xlabel('Threshold')
            #
            # keys = sorted(map(float, binary_results.keys()))
            # colors = ['b', 'g', 'r', 'c', 'm', 'k']
            #
            # for i, k in enumerate(keys):
            #     k = str(k)
            #     r = binary_results[k]
            #
            #     scores = r['scores']
            #     counters = r['counters']
            #     tot = sum(counters.values())
            #     print(counters)
            #
            #     x = i - 0.4 + (0.8 / 6) * 6
            #     ax1.bar(x, scores['global'], width=0.8 / 6,
            #             color=colors[-1], label='Total score' if i == 0 else '')
            #
            #     for j in range(5):
            #         # if j == 5:
            #             # j = 'global'
            #         # else:
            #         kj = str(j)
            #         # if kj not in scores:
            #         #     x = i - 0.4 + (0.8 / 5) * j
            #         #     ax1.bar(x, 0, width=0.8 / 5)
            #
            #         if kj in scores:
            #             x = i - 0.4 + (0.8 / 6) * j
            #             ax1.bar(x, scores[kj], width=0.8 / 6,
            #                     color=colors[j], label='Exit {}'.format(j) if i == 0 else '')
            #
            #         if kj in counters:
            #             x = i - 0.4 + (0.8 / 5) * j
            #             ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
            #                     color=colors[j]) #, label='Exit {}'.format(j) if i == 0 else '')
            #
            #     # for s, c in zip(scores, scores):
            #     #     ax1.bar(i, s, width=0.8 / 5)
            #     #     # ax2.bar(i, c / tot)
            #
            # ax2.set_xticklabels([''] + keys)
            # fig.legend()
            #
            # plt.show()
            #
            # cumulative_results = results['cumulative_results']
            #
            # fig, (ax1, ax2) = plt.subplots(2, sharex=True, num=1)
            # ax1.set_ylim(-0.1, 1)
            # ax2.set_ylim(-0.1, 1)
            #
            # ax1.set_ylabel('Branches scores')
            # ax2.set_ylabel('Branches counters')
            # ax2.set_xlabel('Threshold')
            #
            # keys = sorted(map(float, cumulative_results.keys()))
            # print(keys)
            # colors = ['b', 'g', 'r', 'c', 'k']
            #
            # for i, k in enumerate(keys):
            #     k = str(k)
            #     r = cumulative_results[k]
            #
            #     scores = r['scores']
            #     counters = r['counters']
            #     tot = sum(counters.values())
            #     print(counters)
            #
            #     for j in range(5):
            #         # if j == 5:
            #             # j = 'global'
            #         # else:
            #         kj = str(j)
            #         # if kj not in scores:
            #         #     x = i - 0.4 + (0.8 / 5) * j
            #         #     ax1.bar(x, 0, width=0.8 / 5)
            #         x = i - 0.4 + (0.8 / 5) * j
            #
            #         if kj in scores:
            #             ax1.bar(x, scores[kj], width=0.8 / 5,
            #                     color=colors[j], label='Exit {}'.format(j))
            #
            #         if kj in counters:
            #             ax2.bar(x, counters[kj] / tot, width=0.8 / 5,
            #                     color=colors[j] , label='Exit {}'.format(j))
            #
            #     # for s, c in zip(scores, scores):
            #     #     ax1.bar(i, s, width=0.8 / 5)
            #     #     # ax2.bar(i, c / tot)
            #
            # ax2.set_xticklabels([''] + keys)
            #
            # plt.show()


# @hydra.main(config_path="configs",
#             config_name="config")
# def my_app(cfg: DictConfig) -> None:
#     pass


if __name__ == "__main__":
    my_app()
