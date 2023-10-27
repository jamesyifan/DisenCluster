import numpy as np
import torch
import sklearn
# import sklearn.cluster
from sklearn.cluster import kmeans_plusplus

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    # cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    cosine_dis = cosine.sum(dim=-1).squeeze()
    return cosine_dis

def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    # data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        # data_np = data.detach().cpu().numpy()
        # norm = (data_np**2).sum(axis=1)
        # # init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        # # init = sklearn.cluster.k_means.init(data_np, k, norm, sklearn.utils.check_random_state(None))
        # init, _ = kmeans_plusplus(data_np, k, random_state=sklearn.utils.check_random_state(None))
        # init = torch.tensor(init, requires_grad=True)
        # if num_iter == 0: return init

        # initialize
        # dis_min = float('inf')
        # initial_state_best = None

        dis_max = -float('inf')
        for i in range(20):
            initial_state = initialize(data, k)
            # dis = (data @ initial_state.t()).sum()
            dis = pairwise_cosine(data, initial_state).sum()
            if dis > dis_max:
                dis_max = dis
                init = initial_state

    # if init is None:
    # print('***')
    # print(dis)
    # print(dis_max)
    # print(data)
    # print(k)
    # print('---')
    mu = init.to(data.device)
    # n = data.shape[0]
    # d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
    # iteration = 0
    # tol=1e-4
    # while True:
        #get distances between all data points and cluster centers
        # dist = data @ mu.t()
        dist = pairwise_cosine(data, mu)

        # mu_pre = mu.clone()

        # dist = pairwise_distance(data, mu)
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # print(r.t().unsqueeze(1).shape)
        # print(data.expand(k, *data.shape).shape)
        # print((r.t().unsqueeze(1) @ data.expand(k, *data.shape)).shape)
        # import time
        # time.sleep(10)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean

        mu = new_mu

        # center_shift = torch.sum(
        #     torch.sqrt(
        #         torch.sum((mu - mu_pre) ** 2, dim=1)
        #     ))

        # # increment iteration
        # iteration = iteration + 1

        # if iteration > 500:
        #     break
        # if center_shift ** 2 < tol:
        #     break

    # dist = data @ mu.t()
    dist = pairwise_cosine(data, mu)
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist