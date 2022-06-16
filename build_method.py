def build_local_update_module(args):
    if args.method == 'Fedavg':
        from local_update_method.base import LocalUpdate
    elif args.method == 'GFLN':
        from local_update_method.base_GFLN import LocalUpdate
    else:
        assert False

    return LocalUpdate