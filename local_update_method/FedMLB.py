import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import DatasetSplit


def KD(input_p, input_q, T=1):
    p = F.softmax((input_p / T), dim=1)
    q = F.softmax((input_q / T), dim=1)
    result = ((p * ((p / q).log())).sum()) / len(input_p)

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
        print(p / q)

        print('******************************************************************')
        print('(p/q).log()')
        print((p / q).log())

        print('******************************************************************')
        print('(p*((p/q).log())).sum()')
        print((p * ((p / q).log())).sum())

    return result


class LocalUpdate(object):
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None):
        self.lr = lr
        self.local_epoch = local_epoch
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.args = args
        self.K = len(self.ldr_train)

    def train(self, net):
        model = net
        # train and update
        global_model = copy.deepcopy(model)
        for par in global_model.parameters():
            par.requires_grad = False
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)
        epoch_loss = []
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                out_of_local = model(images, return_feature=True)
                local_features = out_of_local[:-1]
                log_probs = out_of_local[-1]
                log_prob_branch = []
                ce_branch = []
                kl_branch = []
                num_branch = len(local_features)

                ## Compute loss from hybrid branches
                for it in range(num_branch):
                    if self.args.select_level != -1 and self.args.select_level != it:
                        continue
                    this_log_prob = global_model(local_features[it], level=it + 1)
                    this_ce = self.loss_func(this_log_prob, labels)
                    this_kl = KD(this_log_prob, log_probs, self.args.temp)
                    log_prob_branch.append(this_log_prob)
                    ce_branch.append(this_ce)
                    kl_branch.append(this_kl)

                ce_loss = self.loss_func(log_probs, labels)
                loss = self.args.lambda1 * ce_loss + self.args.lambda2 * (
                    sum(ce_branch)) / num_branch + self.args.lambda3 * (sum(kl_branch)) / num_branch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
