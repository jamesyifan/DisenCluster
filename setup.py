from opt import args


def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"
    args.acc = args.nmi = args.ari = args.f1 = 0
    args.dropout = 0.35
    args.n_cluster_iter = 1
    args.clustertemp = 30
    args.mu = 0.0001
    # args.aug = 'add_edge'
    args.aug = None

    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1

        args.K = 5
        args.n_iter = 2
        args.n_layer = 2

    elif args.dataset == 'citeseer':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'sigmoid'
        args.tao = 0.3
        args.beta = 2

        args.K = 3
        args.n_iter = 3
        args.n_layer = 3

    elif args.dataset == 'amap':
        args.t = 4
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 3

        args.K = 10
        args.n_iter = 3
        args.n_layer = 3

    elif args.dataset == 'bat':
        args.t = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.3
        args.beta = 5

        args.K = 50
        args.n_iter = 1
        args.n_layer = 1

    elif args.dataset == 'eat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.7
        args.beta = 5

        args.K = 10
        args.n_iter = 0
        args.n_layer = 0

    elif args.dataset == 'uat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500
        args.activate = 'sigmoid'
        args.tao = 0.7
        args.beta = 5
        
        args.K = 5
        args.n_iter = 3
        args.n_layer = 3

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("confidence: {}".format(args.tao))
    print("focusing factor: {}".format(args.beta))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args
