#!/usr/bin/env python
# coding: utf-8
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from utils import *


def GlobalUpdate(args, device, trainset, testloader, LocalUpdate):
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    for epoch in range(args.global_epochs):
        wandb_dict = {}
        num_of_data_clients = []
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        if (epoch == 0) or (args.participation_rate < 1):
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass
        print(f"This is global {epoch} epoch")

        ## Local training
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user])
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)

        ## Server aggregation
        total_num_of_data_clients = sum(num_of_data_clients)
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i == 0:
                    FedAvg_weight[key] *= num_of_data_clients[i]
                else:
                    FedAvg_weight[key] += local_weight[i][key] * num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients

        ## Update server model
        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ', num_of_data_clients)
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)

        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))
        model.train()
        this_lr *= args.learning_rate_decay

        ## Logging
        wandb_dict[args.mode + "_acc"] = acc_train[-1]
        wandb_dict[args.mode + '_loss'] = loss_avg
        wandb_dict['lr'] = this_lr
        wandb.log(wandb_dict)
        if epoch % args.save_freq == 0:
            if not os.path.exists('{}'.format(args.log_dir)):
                os.makedirs('{}'.format(args.log_dir))
            torch.save({'model_state_dict': model.state_dict(), 'loss_state_dict': criterion.state_dict()},
                       '{}/{}_{}_{}.pth'.format(args.log_dir, args.set, args.arch, epoch + 1)
                       )
