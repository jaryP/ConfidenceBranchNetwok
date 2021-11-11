#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/no_reg'\
 training.device="$DEVICE"  method.prior_w=0 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=ones training.epochs=25

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/ones'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=ones training.epochs=25

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/entropy'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=entropy training.epochs=25

python main.py +dataset=cifar10 +method=bernulli experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/probs'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=probability training.epochs=25

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/no_sample'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=losses \
  method.sample=false method.prior_mode=ones training.epochs=25

#python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/ones_w1\'\
# training.device="$DEVICE"  method.beta=1 experiment.load=true method.joint_type=losses \
#  method.sample=false method.prior_mode=ones training.epochs=25
#
#python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/ones_w0.5\'\
# training.device="$DEVICE"  method.beta=0.5 experiment.load=true method.joint_type=losses \
#  method.sample=false method.prior_mode=ones training.epochs=25
#
#python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/ones_w0.05\'\
# training.device="$DEVICE"  method.beta=0.05 experiment.load=true method.joint_type=losses \
#  method.sample=false method.prior_mode=ones training.epochs=25

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/logits_ones'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=logits \
  method.sample=true method.prior_mode=ones training.epochs=25 method.joint_type=logits method.fix_last_layer=true method.normalize_weights=false

python main.py +dataset=cifar10 +method=bernulli  experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/ablation/cifar10/alexnet/logits_ones'\
 training.device="$DEVICE"  method.prior_w=1 experiment.load=true method.joint_type=losses \
  method.sample=true method.prior_mode=ones training.epochs=25 method.joint_type=logits method.fix_last_layer=true method.normalize_weights=true
