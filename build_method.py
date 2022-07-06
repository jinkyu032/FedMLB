def build_local_update_module(args):
    if args.method == 'FedAvg':
        from local_update_method.base import LocalUpdate
    elif args.method == 'FedMLB':
        from local_update_method.FedMLB import LocalUpdate
    else:
        assert False

    return LocalUpdate
