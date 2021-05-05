from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from torch.optim import Adam
from torch.distributions.continuous_bernoulli import ContinuousBernoulli

from base.evaluators import branches_eval, branches_entropy, branches_binary, \
    branches_mean
from bayesian.posteriors import MatrixEmbedding, LayerEmbeddingBeta, \
    LayerEmbeddingContBernoulli, BayesianHead, BayesianHeads
from models.alexnet import AlexNet
from base.trainer import joint_trainer, output_combiner_trainer, \
    binary_classifier_trainer, posterior_classifier_trainer, \
    binary_posterior_joint_trainer, bayesian_joint_trainer
from base.utils import get_dataset

# from prototipo import branch_lenet


# def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
#     if topk is None:
#         topk = [1, 5]
#
#     if isinstance(topk, int):
#         topk = [topk]
#
#     assert len(expected) == len(predicted)
#     assert predicted.shape[1] >= max(topk)
#
#     res = defaultdict(int)
#     total = len(expected)
#
#     for t, p in zip(expected, predicted):
#         for k in topk:
#             if t in p[:k]:
#                 res[k] += 1
#
#     res = {k: v / total for k, v in res.items()}
#
#     return res


# def joint_training():
#     EPOCHS = 50
#     DEVICE = 'cuda'
#
#     train_set, test_set, eval_set, input_size, classes = \
#         get_dataset('cifar10', 'alexnet')
#
#     train_loader = torch.utils.data.DataLoader(dataset=train_set,
#                                                batch_size=32,
#                                                shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_set,
#                                               batch_size=32,
#                                               shuffle=False)
#
#     predictors = torch.nn.ModuleList()
#     model = AlexNet()
#
#     x, y = next(iter(train_loader))
#     _, outputs = model(x)
#
#     for o in outputs:
#         f = torch.flatten(o, 1)
#         predictors.append(nn.Sequential(nn.Flatten(), nn.Linear(f.shape[-1],
#                                                                 classes)))
#     opt = Adam(chain(model.parameters(), predictors.parameters()), lr=0.001)
#
#     w = [1 / len(predictors)] * len(predictors) + [1]
#     model.to(DEVICE)
#     predictors.to(DEVICE)
#
#     for e in range(EPOCHS):
#         for x, y in train_loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             final_pred, preds = model(x)
#             loss = torch.nn.functional.cross_entropy(final_pred, y,
#                                                      reduction='mean')
#             for i, p in enumerate(preds):
#                 l = torch.nn.functional.cross_entropy(predictors[i](p), y,
#                                                       reduction='mean')
#                 loss += l * w[i]
#
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#         true_labels = []
#         pred_labels = defaultdict(list)
#
#         with torch.no_grad():
#             for x, y in test_loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 final_preds, preds = model(x)
#
#                 true_labels.extend(y.tolist())
#
#                 top_classes = torch.topk(final_preds, final_preds.size(-1))[1]
#                 pred_labels[-1].extend(top_classes.tolist())
#
#                 for i in range(len(preds) - 1):
#                     p = predictors[i](preds[i])
#                     # pred = torch.argmax(pred, -1)
#                     top_classes = torch.topk(p, p.size(-1))[1]
#                     pred_labels[i].extend(top_classes.tolist())
#
#             for i, p in pred_labels.items():
#                 scores = accuracy_score(np.asarray(true_labels),
#                                         np.asarray(p))
#                 print(i, scores[1])
#

# joint_training()

EPOCHS = 20
JOINT_EPOCHS = 25
DEVICE = 'cuda'

train_set, test_set, eval_set, input_size, classes = \
    get_dataset('cifar10', 'alexnet')

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=128,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=128,
                                          shuffle=False)

predictors = torch.nn.ModuleList()
binary_classifiers = torch.nn.ModuleList()

betas = torch.nn.ModuleList()
alphas = torch.nn.ModuleList()

model = AlexNet()

x, y = next(iter(train_loader))
_, outputs = model(x)

for o in outputs:
    f = torch.flatten(o, 1)

for o in outputs:
    inc = o.shape[1]
    bf = torch.flatten(o, 1)
    cv = nn.Conv2d(inc, 128, kernel_size=3)
    # print(o.shape)
    o = cv(o)
    # print(o.shape)
    f = torch.flatten(o, 1)
    # print(f.shape)
    # print()
    binary_classifiers.append(nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
                                            nn.Flatten(),
                                            nn.Linear(f.shape[-1], 1),
                                            nn.Sigmoid()))

    predictors.append(nn.Sequential(nn.Conv2d(inc, 128, kernel_size=3),
                                    nn.Flatten(),
                                    nn.Linear(f.shape[-1], classes)))
    # for o in outputs:
    #     inc = o.shape[1]
    #     cv = nn.Conv2d(inc, 2, kernel_size=3)
    #     o = cv(o)
    #     f = torch.flatten(o, 1)
    betas.append(nn.Sequential(
        # nn.Conv2d(inc, 128, kernel_size=3),
        nn.Flatten(),
        nn.ReLU(),
        # nn.BatchNorm1d(bf.shape[-1]),
        nn.Linear(bf.shape[-1], 1),
        nn.Sigmoid()))

    alphas.append(nn.Sequential(
        # nn.Conv2d(inc, 128, kernel_size=3),
        nn.Flatten(),
        nn.ReLU(),
        # nn.BatchNorm1d(bf.shape[-1]),
        # nn.LayerNorm(),
        nn.Linear(bf.shape[-1], 1),
        nn.ReLU()))

predicts_dict = predictors.state_dict()
predictors.to(DEVICE)
binary_classifiers.to(DEVICE)
model.to(DEVICE)
betas.to(DEVICE)
alphas.to(DEVICE)

# opt = Adam(chain(model.parameters()), lr=0.001)
# best_base_model = model.state_dict()
# #
# best_base_model, scores, best_scores, mean_losses = trainer(model=model,
#                                                             optimizer=opt,
#                                                             train_loader=train_loader,
#                                                             epochs=EPOCHS,
#                                                             scheduler=None,
#                                                             early_stopping=None,
#                                                             test_loader=test_loader,
#                                                             eval_loader=None,
#                                                             device=DEVICE)
# model.load_state_dict(best_base_model)
# print(best_scores)
#
# print('Output combined', best_scores)
# w = torch.nn.Parameter(torch.tensor(
#     [1 / len(predictors)] * len(predictors) + [1]))
# opt = Adam(chain(model.parameters(), predictors.parameters()), lr=0.001)
# _, _, best_scores, _ = output_combiner_trainer(model=model,
#                                                predictors=predictors,
#                                                weights=w,
#                                                optimizer=opt,
#                                                train_loader=train_loader,
#                                                epochs=JOINT_EPOCHS,
#                                                scheduler=None,
#                                                early_stopping=None,
#                                                test_loader=test_loader,
#                                                eval_loader=None,
#                                                device=DEVICE)
# for h in [0.1, 0.25, 0.5, 0.75]:
#     s = branches_entropy(model=model,
#                          threshold=h,
#                          dataset_loader=test_loader,
#                          predictors=predictors,
#                          device=DEVICE)
#     print(h, s)
# print(branches_eval(model, predictors, test_loader, device=DEVICE))
#
# print('Joint training', best_scores)
#
# model.load_state_dict(best_base_model)
# predictors.load_state_dict(predicts_dict)
#
# opt = Adam(chain(model.parameters(),
#                  predictors.parameters()), lr=0.001)
#
# (best_model, best_predictors), _, best_scores, _ = joint_trainer(model=model,
#                                                                  predictors=predictors,
#                                                                  optimizer=opt,
#                                                                  train_loader=train_loader,
#                                                                  epochs=JOINT_EPOCHS,
#                                                                  scheduler=None,
#                                                                  early_stopping=None,
#                                                                  test_loader=test_loader,
#                                                                  eval_loader=None,
#                                                                  device=DEVICE)
# for h in [0.1, 0.25, 0.5, 0.75]:
#     s = branches_entropy(model=model,
#                          threshold=h,
#                          dataset_loader=test_loader,
#                          predictors=predictors,
#                          device=DEVICE)
#     print(h, s)
# print(branches_eval(model, predictors, test_loader, device=DEVICE))
#
# model.load_state_dict(best_model)
# predictors.load_state_dict(best_predictors)
#
# # opt = Adam(chain(model.parameters(),
# #                  predictors.parameters(),
# #                  binary_classifiers.parameters()), lr=0.001, weight_decay=1e-4)
# #
# # _, _, best_scores, _ = binary_classifier_trainer(model=model,
# #                                                  predictors=predictors,
# #                                                  binary_classifiers=binary_classifiers,
# #                                                  optimizer=opt,
# #                                                  train_loader=train_loader,
# #                                                  epochs=JOINT_EPOCHS,
# #                                                  scheduler=None,
# #                                                  early_stopping=None,
# #                                                  test_loader=test_loader,
# #                                                  eval_loader=None,
# #                                                  device=DEVICE)
# # print('binary_classifier_trainer', best_scores)
# print('BEta training', best_scores)
#
# model.load_state_dict(best_base_model)
# predictors.load_state_dict(predicts_dict)

# posteriors = MatrixEmbedding(size=model.branches, distribution='beta')
# posteriors = LayerEmbeddingBeta(beta_layers=betas, alpha_layers=alphas,
#                                 max_clamp=5, min_clamp=0.1)
posteriors = LayerEmbeddingContBernoulli(alpha_layers=betas)
# for h in [0.1, 0.25, 0.5, 0.75]:
#     s = branches_entropy(model=model,
#                          threshold=h,
#                          dataset_loader=test_loader,
#                          predictors=predictors,
#                          device=DEVICE)
#     print(h, s)
# print(branches_eval(model, predictors, test_loader, device=DEVICE))
# exit()
# print(dict(posteriors.named_parameters()).keys())

prior = [Beta(0.1, 2), Beta(0.1, 1), Beta(1, 2), Beta(0.5, 0.5), Beta(2, 1)]
# prior = Beta(0.5, 0.5)
# prior = ContinuousBernoulli(torch.tensor([0.5], device=DEVICE))
for i in np.linspace(0, 1, 50, endpoint=False):
    a = ContinuousBernoulli(torch.tensor([i]))
    print(i, a.entropy().item(), a.mean)

prior = [ContinuousBernoulli(0.1),
         ContinuousBernoulli(0.2),
         ContinuousBernoulli(0.5),
         ContinuousBernoulli(0.8),
         ContinuousBernoulli(0.9)]
# posterior_regularization_trainer

opt = Adam(chain(model.parameters(),
                 predictors.parameters(),
                 posteriors.parameters()),
           lr=0.001)

best_base_model, _, best_scores, _ = posterior_classifier_trainer(model=model,
                                                                  predictors=predictors,
                                                                  posteriors=posteriors,
                                                                  optimizer=opt,
                                                                  prior=prior,
                                                                  prior_w=0.001,
                                                                  energy_w=1e-10,
                                                                  train_loader=train_loader,
                                                                  epochs=5,
                                                                  use_mmd=False,
                                                                  scheduler=None,
                                                                  early_stopping=None,
                                                                  test_loader=test_loader,
                                                                  eval_loader=None,
                                                                  device=DEVICE,
                                                                  cumulative_prior=False)

s, c = branches_binary(model=model,
                       binary_classifiers=posteriors,
                       dataset_loader=test_loader,
                       predictors=predictors,
                       threshold=0.5,
                       device=DEVICE)

print(s, c)

# # for h in [0.1, 0.25, 0.5, 0.75]:
#     s = branches_entropy(model=model,
#                          threshold=h,
#                          dataset_loader=test_loader,
#                          predictors=predictors,
#                          device=DEVICE)
#     print(h, s)


# heads = []
#
# for o in outputs:
#     h = BayesianHead(o, classes, DEVICE)
#     heads.append(h)
#
# predictors = BayesianHeads(heads)
#
# opt = Adam(chain(model.parameters(),
#                  predictors.parameters()),
#            lr=0.001)

# best_base_model, _, best_scores, _ = bayesian_joint_trainer(model=model,
#                                                             predictors=predictors,
#                                                             optimizer=opt,
#                                                             train_loader=train_loader,
#                                                             epochs=10,
#                                                             samples=1,
#                                                             scheduler=None,
#                                                             early_stopping=None,
#                                                             test_loader=test_loader,
#                                                             eval_loader=None,
#                                                             device=DEVICE)

print(branches_eval(model, predictors, test_loader, device=DEVICE))

for h in [0.1, 0.25, 0.5, 0.75]:
    s = branches_entropy(model=model,
                         threshold=h,
                         dataset_loader=test_loader,
                         predictors=predictors,
                         device=DEVICE, samples=2)
    print(h, s)

# a = branches_mean(model, predictors, posteriors, test_loader, device=DEVICE,
#                   topk=[1, 5], c=2.5, samples=5)
# print(a)
