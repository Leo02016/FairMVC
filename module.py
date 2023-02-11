import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from kmeans_pytorch import kmeans
from sklearn.metrics import normalized_mutual_info_score
eps=1e-15


class Weight_Net(nn.Module):
    def __init__(self, hid):
        super(Weight_Net, self).__init__()
        self.linear = nn.Linear(hid, hid)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


def one_hot(data):
    num_unique = np.unique(data).shape[0]
    data = np.array(data)
    new_data = np.zeros((data.shape[0], num_unique))
    label_list = np.unique(data)
    for i in label_list:
        idx = np.argwhere(data == i)
        j = np.argwhere(label_list == i)[0]
        new_data[idx, j] = 1
    return new_data


class MvClustering(nn.Module):
    def __init__(self, d_1=8520, d_2=112, k=5, gpu=0, alpha=0.1, beta=0.01, gamma=1, hid=200, batch_size=100):
        super(MvClustering, self).__init__()
        self.hid = hid
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        self.first_view = nn.Sequential(nn.Linear(d_1, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
                                        nn.Linear(200, self.hid))
        self.second_view = nn.Sequential(nn.Linear(d_2, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
                                         nn.Linear(200, self.hid))
        self.decoder = nn.Sequential(nn.Linear(self.hid, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
                                     nn.Linear(200, self.hid))
        self.v1 = nn.Sequential(nn.Linear(self.hid, 200))
        self.v2 = nn.Sequential(nn.Linear(self.hid, 200))
        self.v = 2
        self.weight_net = Weight_Net(self.hid)
        self.att_layer = Attention_Module(self.hid, batch_size, gpu)

    def forward(self, v1, v2, s, centroid, weighted_unsup=True, train=True, mode='contra'):
        v1_feature = self.first_view(v1)
        v2_feature = self.second_view(v2)
        v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
        v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
        feature = torch.stack([v1_feature, v2_feature])
        raw_feature = torch.cat([v1, v2], dim=1)
        L_c, L_F = 0, 0
        pre_membership = [[] for _ in range(self.v)]
        similarity = [[] for _ in range(self.v)]

        if len(s.shape) == 1:
            s = s.reshape(-1, 1)
        for i in range(self.v):
            similarity[i] = -self.euclidean_dist(centroid[i], feature[i])
            pre_membership[i] = self.additive_smoothing(torch.exp(similarity[i]))
        if self.alpha != 0 and train:
            # # Fairness constraint
            average_ratio = torch.mean(s, dim=0)
            pre_membership = torch.stack(pre_membership)
            # update hard assignment by considering fairness constraint
            membership = [[] for _ in range(self.v)]
            for i in range(self.v):
                cluster_ratio = torch.matmul(pre_membership[i], s) / (torch.sum(pre_membership[i].T, 0) + 1).reshape(-1, 1)
                fairness = torch.matmul((average_ratio - cluster_ratio), (average_ratio - s).T)
                membership[i] = self.additive_smoothing(torch.exp(similarity[i] + self.alpha * fairness))
            membership = torch.stack(membership)
        else:
            membership = torch.stack(pre_membership)
        # KL divergence loss to maximize the mutual agreement of two views
        L_kl = self.kl(membership[0], membership[1]) + self.kl(membership[1], membership[0])
        soft_mem = membership.reshape(self.v * self.k, -1).T
        soft_membership = membership.mean(dim=0).T
        mem = F.one_hot(membership.mean(dim=0).argmax(dim=0), num_classes=self.k).T.double()
        for i in range(self.v):
            distance = torch.exp(-similarity[i])
            L_kl += self.gamma * (distance * mem).mean()
        if self.beta != 0:
            if mode == 'contra':
                L_c= self.contrastive_loss(v1_feature, v2_feature, soft_mem, feature=raw_feature, weighted_unsup=True)
            else:
                L_c = self.simsiam(v1_feature, v2_feature)
        average_ratio = torch.mean(s, dim=0).reshape(-1, 1)
        cluster_ratio = torch.matmul(soft_membership.T, s) / (torch.sum(soft_membership, 0) + 1).reshape(-1, 1)
        L_F = (cluster_ratio - average_ratio).norm(2)
        return [L_kl, self.beta * L_c, self.alpha * L_F], membership

    # def pretrain(self, v1, v2, mode='contra'):
    #     v1_feature = self.first_view(v1)
    #     v2_feature = self.second_view(v2)
    #     v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
    #     v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
    #     raw_feature = torch.cat([v1, v2], dim=1)
    #     if mode == 'contra':
    #         L_c= self.contrastive_loss(v1_feature, v2_feature, 1, feature=raw_feature, weighted_unsup=False)
    #     else:
    #         L_c = self.simsiam(v1_feature, v2_feature)
    #     return L_c

    def contrastive_loss(self, v1, v2, membership, feature=None, weighted_unsup=True):
        # contrastive loss
        size = v1.shape[0]
        if weighted_unsup:
            sim = torch.exp(1 - self.cosine_sim(membership, membership))
        else:
            sim = 1
        similarity = torch.exp(self.cosine_sim(self.v1(v1), self.v2(v2)))
        Loss = torch.mean(torch.log(((sim * similarity).sum(0) + (sim * similarity).sum(1)) / (similarity.diag() * size)))
        return Loss


    def simsiam(self, z1, z2):
        p1, p2 = self.att_layer(self.decoder(z1), self.decoder(z2))  # predictions, n-by-d
        L = self.D(p1, z2)/2 + self.D(p2, z1)/2
        return L

    def D(self, p, z):  # negative cosine similarity
        z = z.detach()  # stop gradient
        p = p/p.norm(2, dim=1, keepdim=True)  # l2-normalize
        return -(p * z).sum(dim=1).mean()


    def compute_centroid(self, v1, v2, s, labels, membership=None):
        v1_feature = self.first_view(v1)
        v2_feature = self.second_view(v2)
        v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
        v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
        feature = torch.stack([v1_feature, v2_feature])
        if membership is None:
            print('Initialize centroid!')
            m1, cluster_centers_v1 = kmeans(X=v1, num_clusters=self.k, distance='euclidean', device=self.device)
            m2, cluster_centers_v2 = kmeans(X=v2, num_clusters=self.k, distance='euclidean', device=self.device)
            one_hot = torch.stack([torch.nn.functional.one_hot(m1), torch.nn.functional.one_hot(m2)]).double()
            one_hot = one_hot.permute(0, 2, 1).to(self.device)
            centroid = (torch.bmm(one_hot, feature).permute(2, 0, 1) / one_hot.sum(dim=2)).permute(1, 2, 0)
        else:
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # centroid: view by # of cluster by dimension
            one_hot = (membership.permute(1, 0, 2) >= membership.max(dim=1)[0]).permute(1, 0, 2).double()
            centroid = torch.bmm(one_hot, feature)
            centroid = ((centroid).permute(2, 0, 1) / (one_hot.sum(dim=2) + 1)).permute(1, 2, 0)
        return centroid

    def kl(self, p, q):
        KL = (p * (p/q).log())
        return KL.sum(0).mean()

    def cosine_sim(self, x1, x2):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / (w1 * w2.t())

    def sim(self, x1, x2, temperature=1):
        return torch.matmul(x1, x2.t()) / temperature

    def additive_smoothing(self, x, eps=1e-5):
        prob = x + eps
        return prob / prob.sum(0).reshape(1, -1)

    def euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def add_dimension_glasso(self, var, dim=0):
        return var.pow(2).sum(dim=dim).add(1e-8).pow(1 / 2.).sum()

    # def split_cluster(self, membership, split_criteria=10):
    #     for i in range(self.v):
    #         cluster_size = membership[i].sum(dim=1)
    #         min_cluster_id = cluster_size.argmin()
    #         if cluster_size[min_cluster_id] < split_criteria:
    #             print('Meet the splitting threshold and start to split the largest cluster into half and merge it with the smallest one...')
    #             max_cluster_id = cluster_size.argmax()
    #             indices = torch.where(membership[i, max_cluster_id, :] == 1)
    #             membership[i, min_cluster_id, indices[:cluster_size[max_cluster_id]]] = 1
    #             membership[i, max_cluster_id, indices[:cluster_size[max_cluster_id]]] = 0
    #     return membership


class Attention_Module(nn.Module):
    def __init__(self, hid=200, n=100, gpu=0):
        super(Attention_Module, self).__init__()
        self.hid = hid
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
        self.similarity_measure = nn.Sequential(nn.Linear(n, n))
        self.w_1 = nn.Sequential(nn.Linear(hid, hid))
        self.w_2 = nn.Sequential(nn.Linear(hid, hid))

    def forward(self, z1, z2):
        C = torch.tanh(torch.matmul(self.similarity_measure(z1.T), z2))
        a = self.w_1(z1)
        b = self.w_2(z2)
        h_1 = torch.tanh(a + torch.matmul(b, C.T))
        h_2 = torch.tanh(torch.matmul(a, C) + b)
        a_1 = F.softmax(h_1, dim=1)
        a_2 = F.softmax(h_2, dim=1)
        return a_1 * z1, a_2 * z2


# class Attention_Module(nn.Module):
#     def __init__(self, hid=200, n=100, gpu=0):
#         super(Attention_Module, self).__init__()
#         self.hid = hid
#         self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
#         self.similarity_measure = nn.Sequential(nn.Linear(hid, hid))
#         self.w_1 = nn.Sequential(nn.Linear(hid, hid))
#         self.w_2 = nn.Sequential(nn.Linear(hid, hid))
#
#     def forward(self, z1, z2):
#         C = torch.tanh(torch.matmul(self.similarity_measure(z1), z2.T))
#         a = self.w_1(z1)
#         b = self.w_2(z2)
#         h_1 = torch.tanh(a + torch.matmul(C, b))
#         h_2 = torch.tanh(torch.matmul(C.T, a) + b)
#         a_1 = F.softmax(h_1, dim=1)
#         a_2 = F.softmax(h_2, dim=1)
#         return a_1 * z1, a_2 * z2


# class MvClustering(nn.Module):
#     def __init__(self, d_1=8520, d_2=112, k=5, gpu=0, alpha=0.1, beta=0.01, hid=200):
#         super(MvClustering, self).__init__()
#         self.hid = hid
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#         self.device = torch.device('cuda' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
#         self.first_view = nn.Sequential(nn.Linear(d_1, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
#                                         nn.Linear(200, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
#                                         nn.Linear(200, self.hid))
#         self.second_view = nn.Sequential(nn.Linear(d_2, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
#                                          nn.Linear(200, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
#                                          nn.Linear(200, self.hid))
#         self.decoder = nn.Sequential(nn.Linear(self.hid, 200), nn.ReLU(inplace=True), nn.BatchNorm1d(200),
#                                      nn.Linear(200, self.hid))
#         self.v1 = nn.Sequential(nn.Linear(self.hid, 200))
#                                 # , nn.ReLU(inplace=True), nn.BatchNorm1d(200), nn.Linear(200, 200))
#         self.v2 = nn.Sequential(nn.Linear(self.hid, 200))
#                                 # , nn.ReLU(inplace=True), nn.BatchNorm1d(200), nn.Linear(200, 200))
#         self.v = 2
#         self.weight_net = Weight_Net(self.hid)
#
#     def forward(self, v1, v2, s, centroid, weighted_unsup=True, train=True, previous_kl=0):
#         v1_feature = self.first_view(v1)
#         v2_feature = self.second_view(v2)
#         v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
#         v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
#         feature = torch.stack([v1_feature, v2_feature])
#         L_c, L_F = 0, 0
#         pre_membership = [[] for _ in range(self.v)]
#         similarity = [[] for _ in range(self.v)]
#         for i in range(self.v):
#             # similarity = 2 * self.cosine_sim(centroid[i], feature[i])
#             similarity[i] = -self.euclidean_dist(centroid[i], feature[i])
#         # compute soft membership without considering fairness constraint
#         for i in range(self.v):
#             pre_membership[i] = self.additive_smoothing(similarity[i])
#         if self.beta != 0 and train:
#             # # Fairness constraint
#             average_ratio = torch.mean(s, dim=0)
#             pre_membership = torch.stack(pre_membership)
#             if len(s.shape) == 1:
#                 s = s.reshape(-1, 1)
#             # update hard assignment considering fairness constraint
#             membership = [[] for _ in range(self.v)]
#             for i in range(self.v):
#                 mem = F.one_hot(pre_membership[i].argmin(dim=0), num_classes=self.k).T.double()
#                 cluster_ratio = torch.matmul(mem, s) / (torch.sum(mem.T, 0) + 1).reshape(-1, 1)
#                 # cluster_ratio = cluster_ratio.T
#                 fairness = (- (cluster_ratio - average_ratio).norm(2, dim=1)).reshape(self.k, -1)
#                 # membership[i] = self.additive_smoothing(similarity[i] + self.beta * fairness)
#                 membership[i] = self.additive_smoothing(similarity[i]) + self.beta * torch.exp(fairness)
#             membership = torch.stack(membership)
#         else:
#             membership = torch.stack(pre_membership)
#         # KL divergence loss to maximize the mutual agreement of two views
#         L_kl = self.kl(membership[0], membership[1]) + self.kl(membership[1], membership[0])
#         # q = []
#         # for i in range(self.v):
#         #     cluster_freq = (membership[i]).sum(dim=1).reshape(-1, 1)
#         #     q.append((membership[i]**2/cluster_freq) / (membership[i]**2/cluster_freq).sum(dim=0))
#         # L_kl = self.kl(membership[0], q[0]) + self.kl(membership[1], q[1])
#         # L_kl = self.kl(q[0], membership[0]) + self.kl(q[1], membership[1])
#         # L_kl2 = 0
#         # weight = 0.001
#         # L_kl = torch.pow(membership[0] - membership[1], 2).sum(0).mean()
#         # for i in range(self.v):
#         #     distance = F.log_softmax(similarity[i], dim=0)
#         #     mem = F.one_hot(distance.argmin(dim=0), num_classes=self.k).T
#         #     # L_kl += -torch.log((distance * mem).sum(0) / (distance * (1-mem)).sum(0)).mean()
#         #     L_kl2 += (-distance * mem + weight * distance * (1 - mem)).mean()
#         #     # L_kl2 += (-distance * mem).mean()
#         #     # L_kl += -torch.log((distance * mem).sum(0) / distance.sum(0)).mean()
#         if self.alpha != 0:
#             # contrastive loss
#             size = v1.shape[0]
#             raw_feature = torch.cat([v1, v2], axis=1)
#             if weighted_unsup:
#                 sim = torch.exp(1 - self.cosine_sim(raw_feature, raw_feature))
#                 # sim = 1 / 2 * (torch.exp(1 - self.cosine_sim(self.weight_net(v1_feature), v2_feature)) +
#                 #                torch.exp(1 - self.cosine_sim(self.weight_net(v2_feature), v1_feature)))
#             else:
#                 sim = 1
#             # similarity = torch.exp(self.cosine_sim(self.v1(v1_feature), self.v2(v2_feature)))
#             # L_c = torch.mean(torch.log(((sim * similarity).sum(0) + (sim * similarity).sum(1)) / (similarity.diag() * size)))
#             L_c = self.simsiam(v1_feature, v2_feature, sim)
#         # # # Fairness constraint
#         average_ratio = torch.mean(s, dim=0)
#         if len(s.shape) == 1:
#             s = s.reshape(-1, 1)
#         # cluster_ratio = torch.sum(torch.matmul(membership, s), dim=0).T / \
#         #                 torch.sum(torch.sum(membership, dim=0), dim=1)
#         # cluster_ratio = cluster_ratio.T
#         # L_F = (cluster_ratio - average_ratio).norm(2)
#         mem = F.one_hot(membership.sum(0).argmin(dim=0), num_classes=self.k).T.double()
#         cluster_ratio = torch.matmul(mem, s) / (torch.sum(mem.T, 0) + 1).reshape(-1, 1)
#         L_F = (cluster_ratio - average_ratio).norm(2)
#         L_lasso = self.add_dimension_glasso(membership[0], 0) + self.add_dimension_glasso(membership[1], 0)
#         # return [L_kl + L_kl2, self.alpha * L_c, self.beta * L_F], membership
#         return [L_kl, self.alpha * L_c, self.beta * L_F, L_lasso * 0.002], membership
#         # return [L_kl, self.alpha * L_c, self.beta * L_F, 0], membership
#
#     def simsiam(self, z1, z2, sim=1):
#         p1, p2 = self.decoder(z1), self.decoder(z2)  # predictions, n-by-d
#         # L = self.D(p1, z2) / 2 + self.D(p2, z1) / 2  # loss
#         L = self.D(p1, z2) / 2 + self.D(p2, z1) / 2 + self.D_negative(p1, z2, sim) / 2 + self.D_negative(p2, z1,sim) / 2
#         return L
#
#     def D(self, p, z):  # negative cosine similarity
#         z = z.detach()  # stop gradient
#         p = p/p.norm(2, dim=1, keepdim=True)  # l2-normalize
#         # z has benn normalized
#         # z = z/z.norm(2, dim=1, keepdim=True)  # l2-normalize
#         return -(p * z).sum(dim=1).mean()
#
#     def D_negative(self, p, z, sim):  # negative cosine similarity
#         z = z.detach()  # stop gradient
#         p = p/p.norm(2, dim=1, keepdim=True)  # l2-normalize
#         # z has been normalized
#         # z = z/z.norm(2, dim=1, keepdim=True)  # l2-normalize
#         mask = torch.eye(z.shape[0], z.shape[0]).bool().to(z.device)
#         return (torch.matmul(p, z.T).masked_fill_(mask, 0) * sim).mean()
#         # return torch.matmul(p, z.T).masked_fill_(mask, 0).mean()
#
#     def compute_centroid(self, v1, v2, s, labels, membership=None):
#         v1_feature = self.first_view(v1)
#         v2_feature = self.second_view(v2)
#         v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
#         v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
#         feature = torch.stack([v1_feature, v2_feature])
#         if membership is None:
#             print('Initialize centroid!')
#             truth = list(labels.data.cpu().numpy())
#             m1, cluster_centers_v1 = kmeans(X=v1, num_clusters=self.k, distance='euclidean', device=self.device)
#             nmi = normalized_mutual_info_score(np.array(truth), m1.cpu().numpy())
#             print('For the first view, nmi:{:.4f}'.format(nmi))
#             m2, cluster_centers_v2 = kmeans(X=v2, num_clusters=self.k, distance='euclidean', device=self.device)
#             nmi2 = normalized_mutual_info_score(np.array(truth), m2.cpu().numpy())
#             print('For the second view, nmi:{:.4f}'.format(nmi2))
#             print('The average nmi:{:.4f}'.format((nmi2 + nmi) / 2))
#             one_hot = torch.stack([torch.nn.functional.one_hot(m1), torch.nn.functional.one_hot(m1)]).double()
#             one_hot = one_hot.permute(0, 2, 1).to(self.device)
#             # fairness evaluation
#             average_ratio = torch.mean(s, dim=0)
#             if len(s.shape) == 1:
#                 s = s.reshape(-1, 1)
#             cluster_ratio = torch.matmul(one_hot.mean(dim=0), s) / (torch.sum(one_hot.mean(0).T, 0) + 1).reshape(-1, 1)
#             # cluster_ratio = cluster_ratio.T
#             # print('The average ratio for the entire data set is {}'.format(average_ratio))
#             # print('The ratio for each cluster is {}'.format(cluster_ratio))
#             print('The fairness score is {}'.format((cluster_ratio - average_ratio).norm(2)))
#             print('\nSize of each cluster:')
#             print(one_hot[0].sum(dim=1).int())
#             print('\nSize of each cluster:')
#             print(one_hot[1].sum(dim=1).int())
#             centroid = (torch.bmm(one_hot, feature).permute(2, 0, 1) / one_hot.sum(dim=2)).permute(1, 2, 0)
#         else:
#             # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
#             # centroid: view by # of cluster by dimension
#             # centroid = torch.bmm(membership, feature) / torch.sum(membership.mean(dim=0), dim=1).reshape(-1, 1)
#             one_hot = (membership.permute(1, 0, 2) >= membership.max(dim=1)[0]).permute(1, 0, 2).double()
#             centroid = torch.bmm(one_hot, feature)
#             centroid = ((centroid).permute(2, 0, 1) / (one_hot.sum(dim=2) + 1)).permute(1, 2, 0)
#         return centroid
#
#     def kl(self, p, q):
#         KL = (p * (p/q).log())
#         # return KL.sum()
#         return KL.sum(0).mean()
#
#     def cosine_sim(self, x1, x2):
#         w1 = x1.norm(p=2, dim=1, keepdim=True)
#         w2 = x2.norm(p=2, dim=1, keepdim=True)
#         return torch.matmul(x1, x2.t()) / (w1 * w2.t())
#
#     def sim(self, x1, x2, temperature=1):
#         return torch.matmul(x1, x2.t()) / temperature
#
#     def additive_smoothing(self, x, eps=1e-5):
#         prob = torch.exp(x) + eps
#         return prob / prob.sum(0).reshape(1, -1)
#
#     def euclidean_dist(self, x, y):
#         n = x.size(0)
#         m = y.size(0)
#         d = x.size(1)
#         assert d == y.size(1)
#         x = x.unsqueeze(1).expand(n, m, d)
#         y = y.unsqueeze(0).expand(n, m, d)
#         return torch.pow(x - y, 2).sum(2)
#
#     def add_dimension_glasso(self, var, dim=0):
#         return var.pow(2).sum(dim=dim).add(1e-8).pow(1 / 2.).sum()
#         # return var.sum(dim=dim).add(1e-8).pow(2).sum()
