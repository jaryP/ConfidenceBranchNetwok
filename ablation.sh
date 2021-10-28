#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar100 +method=bernulli  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar100/resnet20/no_reg'\
 training.device="$DEVICE"  method.prior_w=0 training.batch_size=128 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=ones training.epochs=25

python main.py +dataset=cifar100 +method=bernulli  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar100/resnet20/ones'\
 training.device="$DEVICE"  method.prior_w=0.1 training.batch_size=128 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=ones training.epochs=25


python main.py +dataset=cifar100 +method=bernulli  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar100/resnet20/entropy'\
 training.device="$DEVICE"  method.prior_w=0.1 training.batch_size=128 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=entropy training.epochs=25


python main.py +dataset=cifar100 +method=bernulli experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar100/resnet20/probs'\
 training.device="$DEVICE"  method.prior_w=0.1 training.batch_size=128 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=probability training.epochs=25


python main.py +dataset=cifar100 +method=bernulli  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar100/resnet20/no_sample'\
 training.device="$DEVICE"  method.prior_w=0.1 training.batch_size=128 experiment.load=true method.joint_type=losses \
  method.sample=false method.prior_mode=ones training.epochs=25
