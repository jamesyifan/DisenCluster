import numpy as np
import torch
from opt import args
import torch.nn as nn
import torch.nn.functional as F
from cluster import cluster
from hoscpool import dense_hoscpool

class NeibRoutLayer(nn.Module):
    def __init__(self, num_caps, niter, tau=1.0):
        super(NeibRoutLayer, self).__init__()
        self.k = num_caps
        self.niter = niter
        self.tau = tau

    def forward(self, x, src_trg):
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)
        u = x
        scatter_idx = trg.view(m, 1).expand(m, d)
        for clus_iter in range(self.niter):
            p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
            p = F.softmax(p / self.tau, dim=1)
            scatter_src = (z * p.view(m, k, 1)).view(m, d)
            u = torch.zeros(n, d, device=x.device)
            u.scatter_add_(0, scatter_idx, scatter_src)
            u += x
            # noinspection PyArgumentList
            u = F.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
        return u

# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        x = x.to(torch.float32)
        return torch.mm(x, self.weight) + self.bias

class disen_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, n_iter, n_layer, args):
        super(disen_encoder, self).__init__()
        self.pca = SparseInputLinear(input_dim, hidden_dim)
        # self.pca = nn.Linear(input_dim, hidden_dim)
        conv_ls = []
        self.n_layer = n_layer
        for i in range(n_layer):
            conv = NeibRoutLayer(K, n_iter)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.dropout = args.dropout
    
    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def forward(self, x, src_trg):
        x = self._dropout(F.leaky_relu(self.pca(x)))
        for conv in self.conv_ls:
            x = self._dropout(F.leaky_relu(conv(x, src_trg)))
        # x = self._dropout(self.pca(x))
        # for conv in self.conv_ls:
            # x = self._dropout(conv(x, src_trg))
        return x

class DisenCluster(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, n_num, n_cluster, args):
        super(DisenCluster, self).__init__()
        self.AE1 = nn.Linear(input_dim, hidden_dim)
        self.AE2 = nn.Linear(input_dim, hidden_dim)

        self.SE1 = nn.Linear(n_num, hidden_dim)
        self.SE2 = nn.Linear(n_num, hidden_dim)

        self.dataset = args.dataset

        if self.dataset in ['cora', 'amap']:
            self.ENC1 = disen_encoder(input_dim, hidden_dim, args.K, args.n_iter, args.n_layer, args)
            self.ENC2 = disen_encoder(input_dim, hidden_dim, args.K, args.n_iter, args.n_layer, args)
        else:
            self.ENC1 = disen_encoder(hidden_dim, hidden_dim, args.K, args.n_iter, args.n_layer, args)
            self.ENC2 = disen_encoder(hidden_dim, hidden_dim, args.K, args.n_iter, args.n_layer, args)

        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999).to(args.device)

        self.K = args.K

        self.n_cluster = n_cluster

        self.cluster_temp = args.clustertemp

        self.pooling_type = dense_hoscpool

        self.mu = args.mu

        self.new_ortho = False

        self.init = torch.rand(self.n_cluster, hidden_dim)


        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def forward(self, x, x_1, A, A_1, A_edge, A_edge_1, n_cluster_iter):
        # Z1 = self.activate(self.AE1(x))
        # Z2 = self.activate(self.AE2(x))

        # Z1 = F.normalize(Z1, dim=1, p=2)
        # Z2 = F.normalize(Z2, dim=1, p=2)

        # Z1 = F.normalize(self.AE1(x), dim=1, p=2)
        # Z2 = F.normalize(self.AE2(x), dim=1, p=2)

        # E1 = F.normalize(self.SE1(A), dim=1, p=2)
        # E2 = F.normalize(self.SE2(A), dim=1, p=2)

        Z1 = self.AE1(x)
        Z2 = self.AE2(x_1)

        E1 = self.SE1(A)
        E2 = self.SE2(A_1)

        Z1_E1 = self.alpha * Z1 + (1 - self.alpha) * E1
        Z2_E2 = self.alpha * Z2 + (1 - self.alpha) * E2

        # H1 = self.ENC1(Z1_E1, A_edge)
        # H2 = self.ENC2(Z2_E2, A_edge)
        # print(x)
        # print(x.shape)
        # import time
        # time.sleep(10)
        if self.dataset in ['cora', 'amap']:
            H1 = self.ENC1(x, A_edge)
            H2 = self.ENC2(x_1, A_edge_1)
        else:
            H1 = self.ENC1(Z1_E1, A_edge)
            H2 = self.ENC2(Z2_E2, A_edge)

        n, d = H1.shape

        H1 = F.normalize(self.activate(H1).view(n, self.K, d//self.K), dim=2, p=2).view(n, d)
        H2 = F.normalize(self.activate(H2).view(n, self.K, d//self.K), dim=2, p=2).view(n, d)

        H = (H1 + H2) / 2
        mu, r, dist = cluster(H, self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp)

        _, r1, _ = cluster(H1, self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp)
        _, r2, _ = cluster(H2, self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp)

        return H1, H2, r, r1, r2

    def hoscpool(self, H1, H2, r, A, n_cluster_iter):
        n, d = H1.shape
        H = (H1 + H2) / 2
        H = H.view(n, self.K, d//self.K)
        p_k = []
        init = self.init.view(self.n_cluster, self.K, d//self.K)
        for k in range(self.K):
            # mu_init_k, _, _ = cluster(Z[:,k,:], self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp)

            # mu_init_k, _, _ = cluster(Z[:,k,:], self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp, init=init[:,k,:])
            # mu_k, r_k, dist_k = cluster(Z[:,k,:], self.n_cluster, 1, 1, cluster_temp = self.cluster_temp, init = mu_init_k.detach().clone())

            # mu_k, r_k, dist_k = cluster(Z[:,k,:], self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp, init=init[:,k,:])

            mu_k, r_k, dist_k = cluster(H[:,k,:], self.n_cluster, 1, n_cluster_iter, cluster_temp = self.cluster_temp)

            p_k.append(r_k)

        p_k = torch.cat(p_k, dim=-1)
        H = H.view(n, d)
        _, _, mc, o = self.pooling_type(H, A, p_k, r, self.mu, alpha=1.0, new_ortho=self.new_ortho, mask=None)
        return mc, o 
