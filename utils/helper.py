import torch
import torch.nn as nn

import models

__all__ = ['l2norm', 'get_numclasses', 'count_label_distribution', 'check_data_distribution',
           'check_data_distribution_aug', 'feature_extractor', 'classifier', 'get_model']


def l2norm(x, y):
    z = (((x - y) ** 2).sum())
    return z / (1 + len(x))


class feature_extractor(nn.Module):
    def __init__(self, model, classifier_index=-1):
        super(feature_extractor, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(model.children())[:classifier_index]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class classifier(nn.Module):
    def __init__(self, model, classifier_index=-1):
        super(classifier, self).__init__()
        self.layers = nn.Sequential(
            # stop at conv4
            *list(model.children())[classifier_index:]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def count_label_distribution(labels, class_num: int = 10, default_dist: torch.tensor = None):
    if default_dist != None:
        default = default_dist
    else:
        default = torch.zeros(class_num)
    data_distribution = default
    for idx, label in enumerate(labels):
        data_distribution[label] += 1
    data_distribution = data_distribution / data_distribution.sum()
    return data_distribution


def check_data_distribution(dataloader, class_num: int = 10, default_dist: torch.tensor = None):
    if default_dist != None:
        default = default_dist
    else:
        default = torch.zeros(class_num)
    data_distribution = default
    for idx, (images, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i] += 1
    data_distribution = data_distribution / data_distribution.sum()
    return data_distribution


def check_data_distribution_aug(dataloader, class_num: int = 10, default_dist: torch.tensor = None):
    if default_dist != None:
        default = default_dist
    else:
        default = torch.zeros(class_num)
    data_distribution = default
    for idx, (images, _, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i] += 1
    data_distribution = data_distribution / data_distribution.sum()
    return data_distribution


def get_numclasses(args):
    if args.set in ["CIFAR100"]:
        num_classes = 100
    elif args.set in ["Tiny-ImageNet"]:
        num_classes = 200
    return num_classes


def get_model(args):
    num_classes = get_numclasses(args)
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes, l2_norm=args.l2_norm)
    return model
