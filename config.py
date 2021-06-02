import torch
import torch.nn as nn
from losses import *



import torch
import torch.nn as nn
from losses import *


MNIST_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10),
    "SCE": SCELoss(num_classes=10),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "AEL": AExpLoss(num_classes=10, a=3.5),
    "AUL": AUELoss(num_classes=10, a=3, q=0.1),
    "AGCE": AGCELoss(num_classes=10, a=4, q=0.2),
    "NFL+RCE": NFLandRCE(alpha=1, beta=100, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=100, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=100, num_classes=10),
    "NCEandAGCE": NCEandAGCE(alpha=0, beta=1, num_classes=10, a=4, q=0.2),
    "NCEandAUL": NCEandAUE(alpha=0, beta=1, num_classes=10, a=3, q=0.1),
    "NCEandAEL": NCEandAEL(alpha=0, beta=1, num_classes=10, a=3.5)
}

CIFAR10_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10),
    "SCE": SCELoss(num_classes=10),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "AEL": AExpLoss(num_classes=10, a=2.5),
    "AUL": AUELoss(num_classes=10, a=5.5, q=3),
    "AGCE": AGCELoss(num_classes=10, a=0.6, q=0.6),
    "NFL+RCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
    "NCEandAGCE": NCEandAGCE(alpha=1, beta=4, num_classes=10, a=6, q=1.5),
    "NCEandAUL": NCEandAUE(alpha=1, beta=4, num_classes=10, a=6.3, q=1.5),
    "NCEandAEL": NCEandAEL(alpha=1, beta=4, num_classes=10, a=5)
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100),
    "SCE": SCELoss(num_classes=100),
    # "NLNL": NLNL(train_loader, num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NGCE": NGCELoss(num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "AEL": AExpLoss(num_classes=100, a=2.5),
    "AUL": AUELoss(num_classes=100, a=5.5, q=3),
    "AGCE": AGCELoss(num_classes=100, a=0.6, q=0.6),
    "NFL+RCE": NFLandRCE(alpha=10, beta=1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=100),
    "NCEandAGCE": NCEandAGCE(alpha=10, beta=0.1, num_classes=100, a=1.8, q=3),
    "NCEandAUL": NCEandAUE(alpha=10, beta=0.015, num_classes=100, a=6, q=3),
    "NCEandAEL": NCEandAEL(alpha=10, beta=0.1, num_classes=100, a=1.5)
}

def get_loss_config(dataset, train_loader, loss='CE'):
    if dataset == 'MNIST':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=10)
        elif loss in MNIST_CONFIG:
            return MNIST_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')
    if dataset == 'CIFAR10':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=10)
        elif loss in CIFAR10_CONFIG:
            return CIFAR10_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')
    if dataset == 'CIFAR100':
        if loss == 'NLNL':
            return NLNL(train_loader, num_classes=100)
        elif loss in CIFAR100_CONFIG:
            return CIFAR100_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')


def get_mnist_exp_criterions_and_names(num_classes):
    return list(MNIST_CONFIG.keys()), list(MNIST_CONFIG.values())

def get_cifar10_exp_criterions_and_names(num_classes, train_loader=None):
    return list(CIFAR10_CONFIG.keys()), list(CIFAR10_CONFIG.values())

def get_cifar100_exp_criterions_and_names(num_classes, train_loader):
    return list(CIFAR100_CONFIG.keys()), list(CIFAR100_CONFIG.values())
