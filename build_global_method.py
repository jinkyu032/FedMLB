def build_global_update_module(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.global_method == 'global_avg':
        from global_update_method.base_aggregation import GlobalUpdate
    else:
        assert False
        
    return GlobalUpdate