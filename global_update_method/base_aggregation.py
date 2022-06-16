#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import copy
#from cp
from utils import DatasetSplit
import umap.umap_ as umap
from mpl_toolkits import mplot3d
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
from utils import log_ConfusionMatrix_Umap, log_acc
from utils import calculate_delta_cv,calculate_delta_variance, calculate_divergence_from_optimal,calculate_divergence_from_center
from utils import CenterUpdate
from utils import *
import os

#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def GlobalUpdate(args,device,trainset,testloader,LocalUpdate):
    model = get_model(args)
    model.to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    epoch_loss = []
    weight_saved = model.state_dict()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    ideal_model=copy.deepcopy(model)
    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        if (epoch==0) or (args.participation_rate<1) :
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass 
        print(f"This is global {epoch} epoch")
                        
                        
        if (args.umap==True) and (epoch%args.umap_freq==0):
            if epoch % args.print_freq == 0:                        
                global_acc=log_ConfusionMatrix_Umap(copy.deepcopy(model), testloader, args, wandb_dict,
                                                      name="global model_before local training")
                        
                        
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = LocalUpdate(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                        batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)
            client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size, shuffle=True)
                        

                    
                        
                        
 




        total_num_of_data_clients=sum(num_of_data_clients)        
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        prev_model_weight = copy.deepcopy(model.state_dict())
        current_model_weight = copy.deepcopy(FedAvg_weight)
        
        
        if args.compare_with_center>0:
            if args.compare_with_center ==1:
                idxs=None
            elif args.compare_with_center ==2:
                idxs=[]
                for user in selected_user:
                    idxs+=dataset[user]

            centerupdate = CenterUpdate(args=args,lr = this_lr,iteration_num = len(client_ldr_train)*args.local_epochs,device =device,batch_size=args.batch_size*m,dataset =trainset,idxs=idxs,num_of_participation_clients=m)
            center_weight = centerupdate.train(net=copy.deepcopy(model).to(device))  
            #ideal_weight = centerupdate.train(net=copy.deepcopy(ideal_model).to(device))  
            #ideal_model.load_state_dict(ideal_weight)
            cosinesimilarity_centermodel=calculate_cosinesimilarity_from_center(args, center_weight, current_model_weight, prev_model_weight)
            wandb_dict[args.mode + "_cosinesimilarity_centermodel"] = cosinesimilarity_centermodel
            #divergence_from_central_update = calculate_divergence_from_center(args, center_weight, FedAvg_weight)
            #divergence_from_central_model = calculate_divergence_from_center(args, ideal_weight, FedAvg_weight)
            #wandb_dict[args.mode + "_divergence_from_central_update"] = divergence_from_central_update  
            #wandb_dict[args.mode + "_divergence_from_central_model"] = divergence_from_central_model
        




        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
                                       
                                       
        print(' num_of_data_clients : ',num_of_data_clients)                                   
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)

        if args.analysis:
            ## calculate delta cv
            #delta_cv = calculate_delta_cv(args, copy.deepcopy(model), copy.deepcopy(local_delta), num_of_data_clients)

            ## calculate delta variance
            #delta_variance = calculate_delta_variance(args, copy.deepcopy(local_delta), num_of_data_clients)

            ## Calculate distance from Centralized Optimal Point
            #checkpoint_path = '/data2/geeho/fed/{}/{}/best.pth'.format(args.set, 'centralized')
            #divergence_from_centralized_optimal = calculate_divergence_from_optimal(args, checkpoint_path,
                                                                                   # x_t)
            checkpoint_path = './data/saved_model/fed/CIFAR10/centralized/Fedavg/_best.pth'
            cosinesimilarity=calculate_cosinesimilarity_from_optimal(args, checkpoint_path, current_model_weight, prev_model_weight)
            wandb_dict[args.mode + "_cosinesimilarity"] = cosinesimilarity
            ## Calculate Weight Divergence
            #wandb_dict[args.mode + "_delta_cv"] = delta_cv
            #wandb_dict[args.mode + "_delta_gnsr"] = 1 / delta_cv
            #wandb_dict[args.mode + "_delta_variance"] = delta_variance
            #wandb_dict[args.mode + "_divergence_from_centralized_optimal"] = divergence_from_centralized_optimal

        
                
        if (args.t_sne==True) and (epoch%args.t_sne_freq==0):
            if epoch % args.print_freq == 0:
                model.eval()
                correct = 0
                total = 0
                first=True
                with torch.no_grad():
                    for data in testloader:
                        activation = {}
                        model.layer4.register_forward_hook(get_activation('layer4',activation))
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        if first:
                            features=activation['layer4'].view(len(images),-1)
                            saved_labels=labels
                            first=False
                        else:
                            features=torch.cat((features,activation['layer4'].view(len(images),-1)))
                            saved_labels=torch.cat((saved_labels,labels))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %f %%' % (
                        100 * correct / float(total)))
                acc_train.append(100 * correct / float(total))

            
            
            y_test = np.asarray(saved_labels.cpu())
            tsne = TSNE().fit_transform(features.cpu())
            tx, ty = tsne[:,0], tsne[:,1]
            tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
            ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
            
            plt.figure(figsize = (16,12))


            for i in range(len(classes)):
                y_i = (y_test == i)

                plt.scatter(tx[y_i], ty[y_i], label=classes[i])
            plt.legend(loc=4)
            plt.gca().invert_yaxis()
            #plt.show()
            wandb_dict[args.mode+" t_sne"]=wandb.Image(plt)
            
            
            
            model.train()
        elif (args.umap==False) or (epoch%args.umap_freq!=0):
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
            wandb_dict[args.mode + "_acc"]=acc_train[-1]
                
        else:
            pass

        
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)


        if epoch % args.save_freq == 0:
            if not os.path.exists('{}'.format(args.log_dir)):
                os.makedirs('{}'.format(args.log_dir))
            torch.save({'model_state_dict': model.state_dict(), 'loss_state_dict': criterion.state_dict()},
                       '{}/{}_{}_{}.pth'.format(args.log_dir, args.set, args.arch, epoch + 1)
                       )




# In[ ]:




