import models
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['KD','l2norm','get_numclasses','count_label_distribution','check_data_distribution','check_data_distribution_aug','feature_extractor','classifier','get_model', 'get_optimizer', 'get_scheduler']


def KD(input_p,input_q,T=1):
    p=F.softmax((input_p/T),dim=1)
    q=F.softmax((input_q/T),dim=1)
    result=((p*((p/q).log())).sum())/len(input_p)
    
    if not torch.isfinite(result):
        print('==================================================================')
        print('input_p')
        print(input_p)
        
        print('==================================================================')
        print('input_q')
        print(input_q)
        print('==================================================================')
        print('p')
        print(p)
        
        print('==================================================================')
        print('q')
        print(q)
        
        
        print('******************************************************************')
        print('p/q')
        print(p/q)
        
        print('******************************************************************')
        print('(p/q).log()')
        print((p/q).log())        
        
        print('******************************************************************')
        print('(p*((p/q).log())).sum()')
        print((p*((p/q).log())).sum())            
    
    return result




def l2norm(x,y):
    z= (((x-y)**2).sum())
    return z/(1+len(x))
class feature_extractor(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(feature_extractor, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[:classifier_index]
                )
            def forward(self, x):
                x = self.features(x)
                return x


class classifier(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(classifier, self).__init__()
                self.layers = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[classifier_index:]
                )
            def forward(self, x):
                x = self.layers(x)
                return x

def count_label_distribution(labels,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,label in enumerate(labels):
        data_distribution[label]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images,target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def check_data_distribution_aug(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images, _, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution

def get_numclasses(args):
    if args.set in ['CIFAR10',"MNIST"]:
        num_classes=10
    elif args.set in ["CIFAR100"]:
        num_classes=100
    elif args.set in ["Tiny-ImageNet"]:
        num_classes=200
    return num_classes

def get_model(args):
    num_classes=get_numclasses(args)
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes,l2_norm=args.l2_norm)
    return model


def get_optimizer(args, parameters):
    if args.set=='CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="MNIST":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set=="CIFAR100":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Invalid mode")
        return
    return optimizer



def get_scheduler(optimizer, args):
    if args.set=='CIFAR10':

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** epoch,
        #                         )
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    elif args.set=="CIFAR100":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    else:
        print("Invalid mode")
        return
    return scheduler