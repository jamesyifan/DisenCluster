from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import DisenCluster


if __name__ == '__main__':

    # for n_layer in range(4,5):
    # for n_iter in range(5,7):
    # for K in [1, 3, 5, 10, 30, 50, 75]:
    # for K in [1, 2, 3, 5, 10, 15]:
    # for K in [1, 2, 4, 5, 10, 20]:
    # for K in [20]:
        # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
        # for dataset_name in ["cora", "citeseer", "bat", "eat", "uat"]:
    for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
        # if dataset_name in ['amap', 'uat']:
            # if K == 3:
                # K = 2
            # elif K == 30:
                # K = 20
    # for dataset_name in ["amap"]:
    # for dataset_name in ["bat", "eat", "uat"]:

        # setup hyper-parameter
        args = setup_args(dataset_name)
        # args.n_layer = n_layer
        # args.n_iter = n_iter
        # args.K = K

        # record results
        # file = open("result_iters.csv", "a+")
        file = open("result.csv", "a+")
        # file = open("result_Ks.csv", "a+")
        # file = open("result_layers.csv", "a+")
        # print('n_iter: '+str(args.n_iter), file=file)
        # print('n_layer: '+str(args.n_layer), file=file)
        # print('K: '+str(args.K), file=file)
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for args.seed in range(args.runs):
            # record results

            # fix the random seed
            # setup_seed(args.seed)

            # load graph data
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            # apply the laplacian filtering
            X_filtered = laplacian_filtering(A, X, args.t)

            # augmentation
            if args.aug == 'add_edge':
                A_1 = aug_random_edge(A)
                X_filtered_1 = laplacian_filtering(A_1, X, args.t)
                X_1 = X
            else:
                A_1 = A
                X_filtered_1 = X_filtered
                X_1 = X

            X = torch.tensor(X)

            X_1 = torch.tensor(X_1)

            # test
            args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X, y, cluster_num)

            # build our hard sample aware network
            model = DisenCluster(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num, n_cluster=cluster_num, args=args)

            # adam optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            A_edge = A.nonzero().t().contiguous()

            A_edge_1 = A_1.nonzero().t().contiguous()

            # load data to device
            A, A_edge, model, X, X_filtered, mask = map(lambda x: x.to(args.device), (A, A_edge, model, X, X_filtered, mask))

            A_1, A_edge_1, X_1, X_filtered_1 = map(lambda x: x.to(args.device), (A_1, A_edge_1, X_1, X_filtered_1))
            # A_edge_1 = map(lambda x: x.to(args.device), (A_edge_1))

            n_cluster_iter = args.n_cluster_iter

            if args.dataset in ['cora', 'citeseer']:
                X_ = X.float()
                X_1_ = X_1.float()
            else:
                X_ = X_filtered.float()
                X_1_ = X_filtered_1.float()
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=5, min_lr=0.000001)

            # training
            for epoch in tqdm(range(800), desc="training..."):
                # train mode
                model.train()

                if epoch == 200:
                    n_cluster_iter = 5

                # encoding with Eq. (3)-(5)
                H1, H2, r, r1, r2 = model(X_, X_1_, A, A_1, A_edge, A_edge_1, n_cluster_iter)

                # H_ = (H1 + H2) / 2 
                # H_ = H_.view(-1, args.K, args.dims//args.K)
                # H_abs = H_.norm(dim=-1)
                # B = H_.shape[0]
                # sim_matrix = torch.einsum('bid,bjd->bij', H_, H_) / (1e-8 + torch.einsum('bi,bj->bij', H_abs, H_abs))
                # sim_matrix = torch.exp(sim_matrix)
                # deno = F.normalize(sim_matrix, p=1, dim=-1)
                # i_m = torch.eye(args.K).unsqueeze(0).cuda()
                # orthLoss = torch.abs(deno - i_m).view(B, -1).sum(-1).mean()

                # calculate comprehensive similarity by Eq. (6)
                S = comprehensive_similarity(H1, H2)
                S_r = comprehensive_similarity(r1, r2)

                # calculate hard sample aware contrastive loss by Eq. (10)-(11)
                loss = infoNCE_loss(S, mask, node_num)
                loss_r = infoNCE_loss(S_r, mask, node_num)

                mc, o = model.hoscpool(H1, H2, r, A, n_cluster_iter)

                # loss = loss + mc + o + 5 * orthLoss
                loss = loss + mc + o
                # loss = loss + loss_r + mc + o
                # loss = mc + o + orthLoss
                # loss = orthLoss
                # loss = loss + o
                # loss = loss + mc

                # optimization
                loss.backward()
                optimizer.step()


                # testing and update weights of sample pairs
                if epoch % 1 == 0:
                    # evaluation mode
                    model.eval()

                    # encoding
                    H1, H2, r, r1, r2 = model(X_, X_1_, A, A_1, A_edge, A_edge_1, n_cluster_iter)

                    # calculate comprehensive similarity by Eq. (6)
                    # S = comprehensive_similarity(H1, H2)
                    # S = comprehensive_similarity(r1, r2)

                    # fusion and testing
                    Z = (H1 + H2) / 2
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    scheduler.step(acc)
                    # recording
                    print(acc, args.acc)
                    # print(nmi, args.nmi)
                    if acc >= args.acc:
                    # if nmi >= args.nmi:
                        print(acc)
                        # print(nmi)
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            # file = open("result_iters.csv", "a+")

            save_Z = Z.detach().cpu().numpy() 
            save_P = P
            np.savez('./z_predicty_y'+'_'+dataset_name+'_'+str(args.seed)+'.npz', save_Z, save_P)
            print('representation saved')

            file = open("result.csv", "a+")
            # file = open("result_Ks.csv", "a+")
            # file = open("result_layers.csv", "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        # file = open("result_iters.csv", "a+")
        file = open("result.csv", "a+")
        # file = open("result_Ks.csv", "a+")
        # file = open("result_layers.csv", "a+")
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
