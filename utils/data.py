import torch
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid
import torch.nn as nn


__all__ = ['DatasetSplit', 'DatasetSplitMultiView', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice'
                                                                                                            ]

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class DatasetSplitMultiView(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


def get_dataset(args, trainset, mode='iid'):
    directory = args.client_data + '/' + args.set + '/' + ('un' if args.data_unbalanced==True else '') + 'balanced'
    filepath=directory+'/' + mode + (str(args.dirichlet_alpha) if mode == 'dirichlet' else '') + '_clients' +str(args.num_of_clients) +'.txt'
    check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
    create_new_client_data = not check_already_exist or args.create_client_dataset
    print("create new client data: " + str(create_new_client_data))

    if create_new_client_data == False:
        try:

            dataset = {}
            with open(filepath) as f:
                for idx, line in enumerate(f):
                    dataset = eval(line)
        except:
            print("Have problem to read client data")

    if create_new_client_data == True:
        if mode == 'iid':
            dataset = cifar_iid(trainset, args.num_of_clients)
        elif mode == 'skew1class':
            dataset = cifar_noniid(trainset, args.num_of_clients)
        elif mode == 'dirichlet':
            if args.data_unbalanced==True:
                dataset = cifar_dirichlet_unbalanced(trainset, args.num_of_clients, alpha=args.dirichlet_alpha)
            else:
                dataset = cifar_dirichlet_balanced(trainset, args.num_of_clients, alpha=args.dirichlet_alpha)
        else:
            print("Invalid mode ==> please select in iid, skew1class, dirichlet")
            return
        try:
            os.makedirs(directory, exist_ok=True)
            with open(filepath, 'w') as f:
                print(dataset, file=f)

        except:
            print("Fail to write client data at " + directory)

    return dataset



class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
