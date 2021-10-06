#!/usr/bin/env bash

DATASET=$1
DEVICE=$2


case $DATASET in

  mnist)
  python main.py +dataset=mnist +method=joint experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint +method.train_weights=true experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint_training' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint experiment=base +method.joint_type=losses +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint_losses' training.device="$DEVICE"

#  python main.py +dataset=mnist +method=bernulli_fixed experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/bernulli_fixed' training.device="$DEVICE"
    ;;
#
#  cifar10)
#  python main.py +dataset=cifar10 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/' +attacks=ps training.device="$DEVICE"
#    ;;
#
#  tinyimagenet)
#  python main.py +dataset=tinyimagenet experiment=base +model=resnet18 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs/tinyimagenet/' +attacks=ps_tiny training.device="$DEVICE"
#    ;;
#
#  cub200)sgd_momentum
#  python main.py +dataset=cub200 experiment=base +model=resnet34 optimizer=sgd_momentum +training=cub200 hydra.run.dir='./outputs/cub200/' +attacks=ps training.device="$DEVICE"
#    ;;

  *)
  echo -n "Unrecognized dataset"

esac
