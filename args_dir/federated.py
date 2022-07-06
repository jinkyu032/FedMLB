import argparse

# from configs import parser as _parser


USABLE_TYPES = set([float, int])


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def produce_override_string(args, override_args):
    lines = []
    for v in override_args:
        if v != "multigpu":
            v_arg = getattr(args, v)
            if type(v_arg) in USABLE_TYPES:
                lines.append(v + ": " + str(v_arg))
            else:
                lines.append(v + ": " + f'"{str(v_arg)}"')
        else:
            lines.append("multigpu: " + str(args.multigpu))

    return "\n# ===== Overrided ===== #\n" + "\n".join(lines)


args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--mode", help="data setting to use", default="iid"
    )
    parser.add_argument(
        "--data_unbalanced", action='store_true', help="create unbalanced client data"
    )
    parser.add_argument(
        "--project", help="data setting to use", default="federated_learning"
    )
    parser.add_argument(
        "--method", help="method to use", default="FedMLB"
    )
    parser.add_argument(
        "--loss", help="loss to use", default="CE"
    )
    parser.add_argument(
        "--return_feature", action='store_true', help="Do you need to make model forward return feature?"
    )
    parser.add_argument(
        "--global_method", help="aggregation method to use", default="global_avg"
    )
    parser.add_argument(
        "--additional_experiment_name", help="specific information for logging", default=""
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--cuda_visible_device",
        default=0,
        type=float,
        metavar="N",
        help="which GPU to use",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--create_client_dataset",
        action="store_true",
        help="Whether or not to make new client dataset",
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )

    ## Training Config
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--global_epochs",
        default=1000,
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
    parser.add_argument(
        "-b",
        "--batch_size",
        default=50,
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
        default=1e-3,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--ld",
        "--learning_rate_decay",
        default=0.998,
        type=float,
        metavar="LD",
        help="learning rate decay (default: 1.0)",
        dest="learning_rate_decay",
    )
    parser.add_argument(
        "--gr_clipping_max_norm",
        default=10,
        type=float,
        metavar="N",
        help="gradient clippling max norm",
    )

    ## Dataset
    parser.add_argument("--set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "--dirichlet_alpha",
        default=-0.3,
        type=float,
        metavar="N",
        help="federated dirichlet alpha",
    )
    parser.add_argument(
        "--data", help="path to dataset base directory", default="./data"
    )
    parser.add_argument(
        "--client_data", help="path to client dataset base directory", default="./client_data"
    )
    parser.add_argument("--num-classes", default=10, type=int)

    ## Eval
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,

        help="print frequency (default: 1)",
    )
    parser.add_argument(
        "--save-freq",
        default=1000,
        type=int,
        help="save frequency (default: 1000)",
    )

    # FedMLB
    parser.add_argument(
        "--temp",
        default=1,
        type=float,
        metavar="N",
        help="temperature",
    )
    parser.add_argument(
        "--select_level", default=-1, type=int, help="Decide where to distill"
    )
    parser.add_argument(
        "--lambda1",
        default=1,
        type=float,
        metavar="N",
        help="Weight for CE loss of main pathway",
    )
    parser.add_argument(
        "--lambda2",
        default=1,
        type=float,
        metavar="N",
        help="Weight for CE loss of hybrid pathways",
    )
    parser.add_argument(
        "--lambda3",
        default=1,
        type=float,
        metavar="N",
        help="Weight for KD loss of hybrid pathways",
    )
    parser.add_argument(
        "--l2_norm",
        action="store_true",
        help="l2_norm classifier",
    )

    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
