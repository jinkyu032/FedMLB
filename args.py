import argparse
import sys
import yaml
from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--mode", help="data setting to use", default="centralized"
    )
    parser.add_argument(
        "--project", help="data setting to use", default="federated_learning"
    )
    parser.add_argument(
        "--method", help="method to use", default="Fedavg"
    )
    parser.add_argument(
        "--global_method", help="aggregation method to use", default="global_avg"
    )
    parser.add_argument(
        "--additional_experiment_name", help="specific information for logging", default=""
    )
    '''
    parser.add_argument(
        "--centralized",
        action="store_true",
        help="Whether or not to train in centralized setting",
    )
    parser.add_argument(
        "--federated_iid",
        action="store_true",
        help="Whether or not to train in federated setting",
    )
    parser.add_argument(
        "--federated_skew1class",
        action="store_true",
        help="Whether or not to train in 1class skewed Non-iid federated setting",
    ) 
    parser.add_argument(
        "--federated_dirichlet",
        action="store_true",
        help="Whether or not to train in dirichlet Non-iid federated setting",
    )
    '''
    parser.add_argument(
        "--create_client_dataset",
        action="store_true",
        help="Whether or not to make new client dataset",
    )
    parser.add_argument(
        "--global_epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--centralized_epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--local_epochs",
        default=5,
        type=int,
        metavar="N",
        help="number of local epochs to run on each global epoch",
    )    
    parser.add_argument(
        "--dirichlet_alpha",
        default=-1,
        type=float,
        metavar="N",
        help="federated dirichlet alpha",
    )
    
    parser.add_argument(
    "--cuda_visible_device",
    default=0,
    type=float,
    metavar="N",
    help="which GPU to use",
    )
    parser.add_argument(
    "--alpha",
    default=0,
    type=float,
    metavar="N",
    help="federated activation regularization hyperparameter alpha",
    )
    parser.add_argument(
        "--alpha_mul_epoch",
        action="store_true",
        help="Whether or not to multiplicate alpha with epoch",
    )    
    parser.add_argument(
        "--alpha_divide_epoch",
        action="store_true",
        help="Whether or not to divide alpha by epoch",
    )    
    parser.add_argument(
    "--epsilon",
    default=0.1,
    type=float,
    metavar="N",
    help="Used in LSC Loss",
    )
    
    
    parser.add_argument(
        "--data", help="path to dataset base directory", default="./data"
    )
    parser.add_argument(
        "--client_data", help="path to client dataset base directory", default="./client_data"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )



    parser.add_argument(
        "-b",
        "--batch_size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--momentum", default=0.0, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--ld",
        "--learning_rate_decay",
        default=1.0,
        type=float,
        metavar="LD",
        help="learning rate decay (default: 1.0)",
        dest="learning_rate_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 1)",
    )
    parser.add_argument("--num-classes", default=10, type=int)

    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    
    parser.add_argument(
        "--num_of_clients",
        default=100,
        type=int,
        metavar="N",
        help="number of total clients",
    )
    parser.add_argument(
        "--participation_rate",
        default=0.1,
        type=float,
        metavar="N",
        help="number of participation rate at each global epoch",
    )
    '''

    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    '''



    '''
    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )


    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    '''


    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
