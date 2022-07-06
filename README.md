## Training FedMLB

### CIFAR100, Dirichlet 0.6, 100 clients, 5% participation


```
python federated_train.py --cuda_visible_device 2 --config configs/cifar_Fedavg.yaml --data ./data --mode dirichlet --dirichlet_alpha 0.6  --centralized_epochs 0 --global_epochs 1000 --local_epochs 5 --epsilon 0.1 --momentum 0 --lr 0.1 --learning_rate_decay 0.998 --weight_decay 1e-3 --seed 0 --set CIFAR100 --arch ResNet18_FedMLB --workers 8  --alpha 0 --additional_experiment_name "FedAvg balanced_p_rate_0.05_sync" --participation_rate=0.05 --learning_rate_decay 0.998 --num_of_clients=100  --method GFLN --batch_size=50
```