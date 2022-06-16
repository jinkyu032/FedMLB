import torch
import torch.nn as nn


__all__ = ['IL','CE','IL_negsum', 'DeepInversionFeatureHook']

class IL_negsum():
    def __init__(self,device,mean=True,gap=0.5,abs_thres=True):
        self.device=device
        self.mean=mean
        self.gap=gap
        self.abs_thres=abs_thres
        self.top_num=1
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        neg_losses=sigmoid*(1-onehot)
        val,idx=neg_losses.topk(self.top_num)
        print("sigmoid",sigmoid[0])
        #print(neg_losses[0])
        #print(onehot[0])
        '''
        extend_p_of_answer=(p_of_answer*self.gap).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
            print("sigmoid",sigmoid[0])
            print("bigger_than_answer",bigger_than_answer[0])
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer))*(1-onehot)'''
        pos=(((1-p_of_answer)**2))*self.top_num
        neg=(((val**2)).sum(dim=1))##((bigger_than_answer.sum(dim=1)+1e-10))
        s=pos+neg
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s


class IL():
    def __init__(self,device,mean=True,gap=0.5,abs_thres=True):
        self.device=device
        self.mean=mean
        self.gap=gap
        self.abs_thres=abs_thres
    def __call__(self,outputs,labels):
        l=len(labels)
        sigmoid=torch.sigmoid(outputs)
        onehot=(torch.eye(10)[labels]).to(self.device)
        p_of_answer=(sigmoid*onehot).sum(axis=1)

        extend_p_of_answer=(p_of_answer*self.gap).unsqueeze(dim=1).expand(outputs.shape)
        if self.abs_thres==True:
            bigger_than_answer=(sigmoid>(self.gap))*(1-onehot)
            #print("sigmoid",sigmoid[0])
            #print("bigger_than_answer",bigger_than_answer[0])
        else:
            bigger_than_answer=(sigmoid>(extend_p_of_answer))*(1-onehot)
        pos=(((1-p_of_answer)**2))
        neg=(((sigmoid*bigger_than_answer)**2).sum(dim=1))/9##((bigger_than_answer.sum(dim=1)+1e-10))
        s=pos+neg
        
        s=s.sum()
        if self.mean==True:
            s/=l
        return s


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


CE=nn.CrossEntropyLoss()