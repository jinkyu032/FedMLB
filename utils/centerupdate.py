import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit,IL,IL_negsum
import torch


####### batch_size는 client batch_size의 num_of_participation_clients배가 되어야 할 것.
class CenterUpdate(object):
    def __init__(self, args, lr, iteration_num, device, batch_size, dataset=None, idxs=None,num_of_participation_clients = 10):
        self.lr=lr
        self.iteration_num=iteration_num
        self.device=device
        if args.loss=='CE':
            self.loss_func=nn.CrossEntropyLoss()
        elif args.loss in ('IL','Individual_loss'):
            self.loss_func=IL(device=device,gap=args.thres,abs_thres=args.abs_thres)
        elif args.loss=='IL_negsum':
            self.loss_func=IL_negsum(device=device,gap=args.thres,abs_thres=args.abs_thres)
        self.selected_clients = []
        
        
        #### idxs가 None이면, trainset에 있는 전체의 data를 이용한다.
        if idxs ==None:
            self.idxs  = range(len(dataset))
            
        #### idxs가 None이 아니고 이번 round에 쓰인 user들의 indicies들의 tuple,예를 들어 (user1,user2,...)의 경우 각 client들의 
        #### data들의 합집합을 사용한다. 
        else:
            self.idxs=idxs           
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs), batch_size=batch_size, shuffle=True)
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None):
        model = net
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        count = 0
        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count ==self.iteration_num:
                    break
                count+=1
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
            if count ==self.iteration_num:
                break 
        return net.state_dict()
