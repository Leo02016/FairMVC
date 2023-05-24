import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kmeans_pytorch import kmeans
eps=1e-15


def balance_score(y_hat, r):
    return (torch.mm(y_hat, r).min(dim=1)[0]/torch.sum(y_hat, dim=1)).min()


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
        self.decoder = nn.Sequential(nn.Linear(self.hid, 500), nn.ReLU(inplace=True), nn.BatchNorm1d(500),
                                     nn.Linear(500, self.hid))
        self.v1 = nn.Sequential(nn.Linear(self.hid, 200))
        self.v2 = nn.Sequential(nn.Linear(self.hid, 200))
        self.v = 2
        self.att_layer = Attention_Module(self.hid, batch_size, gpu)
        self.temperature = 1
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

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
            average_ratio = torch.mean(s, dim=0).T
            pre_membership = torch.stack(pre_membership)
            # update hard assignment by considering fairness constraint
            membership = [[] for _ in range(self.v)]
            soft_membership = pre_membership.mean(dim=0)
            cluster_ratio = torch.matmul(soft_membership, s) / (torch.sum(soft_membership.T, 0) + 1).reshape(-1, 1)
            expand_cluster_ratio = cluster_ratio.unsqueeze(1).expand(self.k, s.shape[0], s.shape[1])
            # matrix element-wise multiplication:
            # input_1.shape= n X k X d, input_2.shape= k X d, output_shape = n X k X d
            Fairness = torch.mul((expand_cluster_ratio-s).permute((1, 0, 2)), cluster_ratio-average_ratio).sum(dim=2)
            for i in range(self.v):
                membership[i] = self.additive_smoothing(torch.exp(similarity[i] + self.alpha * Fairness.T))
            membership = torch.stack(membership)
        else:
            membership = torch.stack(pre_membership)
        # KL divergence loss to maximize the mutual agreement of two views
        L_kl = self.gamma * (self.kl(membership[0], membership[1]) + self.kl(membership[1], membership[0]))
        soft_membership = membership.mean(dim=0).T
        mem = F.one_hot(membership.mean(dim=0).argmax(dim=0), num_classes=self.k).T.double()
        for i in range(self.v):
            distance = torch.exp(-similarity[i])
            L_kl += (distance * mem).mean()
        if self.beta != 0:
            if mode == 'contra':
                L_c= self.contrastive_loss2(v1_feature, v2_feature, membership, feature=raw_feature, weighted_unsup=True)
            else:
                L_c = self.simsiam(v1_feature, v2_feature)
        # Fairness constraint
        average_ratio = torch.mean(s, dim=0).reshape(-1, 1).T
        cluster_ratio = torch.matmul(soft_membership.T, s) / (torch.sum(soft_membership, 0) + 1).reshape(-1, 1)
        L_F = (cluster_ratio - average_ratio).norm(2)
        return [L_kl, self.beta * L_c,  self.alpha * L_F], membership

    def pretrain(self, v1, v2, mode='contra'):
        v1_feature = self.first_view(v1)
        v2_feature = self.second_view(v2)
        v1_feature = v1_feature / v1_feature.norm(p=2, dim=1, keepdim=True)
        v2_feature = v2_feature / v2_feature.norm(p=2, dim=1, keepdim=True)
        raw_feature = torch.cat([v1, v2], dim=1)
        if mode == 'contra':
            L_c= self.contrastive_loss(v1_feature, v2_feature, 1, feature=raw_feature, weighted_unsup=False)
        else:
            L_c = self.simsiam(v1_feature, v2_feature)
        return L_c

    def contrastive_loss(self, v1, v2, membership, feature=None, weighted_unsup=True):
        # contrastive loss
        size = v1.shape[0]
        if weighted_unsup:
            sim = torch.exp(1 - self.cosine_sim(feature, feature))
            # sim = torch.exp(1 - self.cosine_sim(membership, membership))
        else:
            sim = 1
        similarity = torch.exp(self.cosine_sim(self.v1(v1), self.v2(v2)))
        Loss = torch.mean(torch.log(((sim * similarity).sum(0) + (sim * similarity).sum(1)) / (similarity.diag() * size)))
        return Loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def contrastive_loss2(self, v1, v2, membership, feature=None, weighted_unsup=True):
        N = 2 * v1.shape[0]
        z = torch.cat((v1, v2), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        membership = membership.permute(2,0,1).reshape(-1,self.k)
        weight_sim = torch.exp(1 - self.cosine_sim(membership, membership))
        weight_i_j = torch.diag(weight_sim, self.batch_size)
        weight_j_i = torch.diag(weight_sim, self.batch_size)
        positive_weight = torch.cat((weight_i_j, weight_j_i), dim=0).reshape(N, 1)
        negative_weight = weight_sim[self.mask].reshape(N, -1)
        weight = torch.cat((positive_weight, negative_weight), dim=1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) * weight.detach()
        # logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def simsiam(self, z1, z2):
        # p1, p2 = self.decoder(z1), self.decoder(z2)  # predictions, n-by-d
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

