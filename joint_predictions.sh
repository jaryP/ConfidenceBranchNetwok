#!/usr/bin/env bash

DATASET=$1
DEVICE=$2


case $DATASET in
  cifar10)
  python main.py +dataset=cifar10 +method=joint_prediction1 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint_prediction1' training.device="$DEVICE"
  python main.py +dataset=cifar10 +method=joint_prediction2 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint_prediction2' training.device="$DEVICE"
  python main.py +dataset=cifar10 +method=joint_prediction3 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint_prediction3' training.device="$DEVICE"
    ;;

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
