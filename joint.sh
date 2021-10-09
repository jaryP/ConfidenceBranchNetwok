#!/usr/bin/env bash

DATASET=$1
DEVICE=$2


case $DATASET in

  mnist)
  python main.py +dataset=mnist +method=joint experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint2 experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint2' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint3 experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint3' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint4 experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint4' training.device="$DEVICE"
  python main.py +dataset=mnist +method=joint5 experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/joint5' training.device="$DEVICE"

#  python main.py +dataset=mnist +method=bernulli_fixed experiment=base +model=alexnet optimizer=sgd_momentum +training=mnist training.epochs=30 hydra.run.dir='./outputs/mnist/bernulli_fixed' training.device="$DEVICE"
    ;;
#
  cifar10)
  python main.py +dataset=cifar10 +method=joint experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint' training.device="$DEVICE"
  python main.py +dataset=cifar10 +method=joint2 experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint2' training.device="$DEVICE"
  python main.py +dataset=cifar10 +method=joint3 experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/joint3' training.device="$DEVICE"
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
