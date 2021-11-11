#!/usr/bin/env bash

DATASET=$1
MODEL=$2
DEVICE=$3

case $DATASET in
  svhn)
    case $MODEL in
    alexnet)
      python main.py +dataset=svhn +method=joint_alexnet experiment=base +model=alexnet optimizer=sgd_momentum +training=svhn hydra.run.dir='./outputs/svhn/alexnet/joint' training.device="$DEVICE" experiment.load=true
      python plots.py  './outputs/svhn/alexnet/joint'
    ;;
    resnet20)
      python main.py +dataset=svhn +method=joint_resnet20 experiment=base +model=resnet20 optimizer=sgd_momentum +training=svhn hydra.run.dir='./outputs/svhn/resnet20/joint' training.device="$DEVICE" experiment.load=true
      python plots.py  './outputs/svhn/resnet20/joint'
    ;;
    vgg11)
      python main.py +dataset=svhn +method=joint_vgg11 experiment=base +model=vgg11 optimizer=sgd_momentum +training=svhn hydra.run.dir='./outputs/svhn/vgg11/joint' training.device="$DEVICE" experiment.load=true
      python plots.py  './outputs/svhn/vgg11/joint'
    ;;
    *)
      echo -n "Unrecognized model"
    esac
  ;;
  cifar10)
    case $MODEL in
    alexnet)
      python main.py +dataset=cifar10 +method=joint_alexnet experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/alexnet/joint' training.device="$DEVICE" experiment.load=true
      python plots.py './outputs/cifar10/alexnet/joint'
    ;;
    resnet20)
      python main.py +dataset=cifar10 +method=joint_resnet20 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/resnet20/joint' training.device="$DEVICE" experiment.load=true
      python plots.py './outputs/cifar10/resnet20/joint'
    ;;
    vgg11)
      python main.py +dataset=cifar10 +method=joint_vgg11 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/vgg11/joint' training.device="$DEVICE" experiment.load=true
      python plots.py  './outputs/cifar10/vgg11/joint'
    ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  cifar100)
    case $MODEL in
      alexnet)
        python main.py +dataset=cifar100 +method=joint_alexnet experiment=base +model=alexnet optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/cifar100/alexnet/joint' training.device="$DEVICE" experiment.load=true
        python plots.py './outputs/cifar100/alexnet/joint'
        ;;
      resnet20)
        python main.py +dataset=cifar100 +method=joint_resnet20 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/cifar100/resnet20/joint' training.device="$DEVICE" experiment.load=true
        python plots.py  './outputs/cifar100/resnet20/joint'
        ;;
      vgg11)
        python main.py +dataset=cifar100 +method=joint_vgg11 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar100 hydra.run.dir='./outputs/cifar100/vgg11/joint' training.device="$DEVICE" experiment.load=true
        python plots.py './outputs/vgg11/resnet20/joint'
        ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  *)
  echo -n "Unrecognized dataset"
esac
