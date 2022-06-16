import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit,IL,IL_negsum
import torch
from local_update_method.global_and_online_model import *



class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        if args.loss=='CE':
            self.loss_func=nn.CrossEntropyLoss()
        elif args.loss in ('IL','Individual_loss'):
            self.loss_func=IL(device=device,gap=args.thres,abs_thres=args.abs_thres)
        elif args.loss=='IL_negsum':
            self.loss_func=IL_negsum(device=device,gap=args.thres,abs_thres=args.abs_thres)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None):
        #model=dual_model(self.args,net,net)
        model = net
        # train and update
        optimizer = optim.SGD(model.parameters(), lr=self.lr,momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
