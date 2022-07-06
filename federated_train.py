import os
import random

import numpy as np
import torch
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets as local_datasets
from args_dir.federated import args
from build_global_method import build_global_update_module
from build_method import build_local_update_module

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)

experiment_name = args.set + "_" + args.mode + (
    str(args.dirichlet_alpha) if args.mode == 'dirichlet' else "") + "_" + args.method + (
                      "_" + args.additional_experiment_name if args.additional_experiment_name != '' else '')
print(experiment_name)
wandb_log_dir = os.path.join('./', experiment_name)
if not os.path.exists('{}'.format(wandb_log_dir)):
    os.makedirs('{}'.format(wandb_log_dir))
wandb.init(entity='federated_learning', project=args.project,
           group=args.mode + (str(args.dirichlet_alpha) if args.mode == 'dirichlet' else ""), job_type=args.method + (
        "_" + args.additional_experiment_name if args.additional_experiment_name != '' else '')
           , dir=wandb_log_dir)
wandb.run.name = experiment_name
wandb.run.save()
wandb.config.update(args)

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Build Dataset
if args.set in ['CIFAR100']:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform_train = transforms.Compose(
        [transforms.RandomRotation(10),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize])

    if args.set == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                                download=True, transform=transform_test)
        classes = tuple(str(i) for i in range(100))

elif args.set in ['Tiny-ImageNet']:
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # Add RandomRotation
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    trainset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.data, 'tiny_imagenet'),
        split='train',
        transform=transform_train
    )
    testset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.data, 'tiny_imagenet'),
        split='test',
        transform=transform_train
    )
    classes = tuple(str(i) for i in range(100))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

LocalUpdate = build_local_update_module(args)
global_update = build_global_update_module(args)
global_update(args=args, device=device, trainset=trainset, testloader=testloader, LocalUpdate=LocalUpdate)
