from data_loader import *
import time
from module import MvClustering
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import argparse
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu>=0 else 'cpu')
    args.device = device
    if args.data == 'credit':
        dataset = load_uci_credit(use_mask=args.miss, index=args.index, missing_ratio=0.5, purturbed=args.purturbed)
        batch_size = 5000
        d_1 = dataset.data.shape[1]
        d_2 = dataset.data2.shape[1]
        k = dataset.k
    elif args.data == 'mnist':
        dataset = load_mnist(5000, 0.1, use_mask=args.miss, index=args.index, missing_ratio=0.25)
        batch_size = 5000
        d_1 = dataset.data.shape[1]
        d_2 = dataset.data2.shape[1]
        k = dataset.k
    elif args.data == 'bank':
        dataset = load_bank()
        batch_size = 5000
        d_1 = dataset.data.shape[1]
        d_2 = dataset.data2.shape[1]
        k = dataset.k
    elif args.data == 'zafar':
        dataset = load_zafar(idx=args.index, use_mask=args.miss, missing_ratio=0.25, purturbed=args.purturbed)
        batch_size = 10000
        d_1 = dataset.data.shape[1]
        d_2 = dataset.data2.shape[1]
        k = dataset.k
    elif args.data == 'xrmb':
        dataset = load_xrmb(6, 3000)
        batch_size = 10000
        d_1 = dataset.data.shape[1]
        d_2 = dataset.data2.shape[1]
        k = dataset.k
    else:
        raise(NameError('Please {} data set is not support yet'.format(args.data)))
    if batch_size > dataset.data.shape[0]:
        batch_size = dataset.data.shape[0]
    model = MvClustering(d_1=d_1, d_2=d_2, k=k, gpu=0, alpha=args.alpha, beta=args.beta, hid=args.hidden,
                         gamma=args.gamma, batch_size=batch_size).to(device)
    model = model.double()
    model.learning_rate = args.lr
    model.epoch = args.epoch
    model.batch_size = batch_size
    print('The size of training samples: {}\nBatch size: {}'.format(len(dataset), model.batch_size))
    print('Initial learning rate: {}'.format(args.lr))
    print('Size of hidden representation: {}'.format(args.hidden))
    print('alpha = {}, beta = {}'.format(args.alpha, args.beta))
    train_model(model, dataset, args)


def batch(iterable, n=1):
    l = len(iterable)
    iterable.shuffle()
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def train_model(model, train_loader, args):
    epoch = 0
    cur_iter = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    start_time = time.time()
    for train_sample in batch(train_loader, train_loader.data.shape[0]):
        v1 = torch.tensor(train_sample['data']).double().to(model.device)
        labels = torch.tensor(train_sample['label']).double().to(model.device)
        v2 = torch.tensor(train_sample['data2']).double().to(model.device)
        s = torch.tensor(train_sample['sensitive']).double().to(model.device)
        centroid = model.compute_centroid(v1, v2, s, labels, membership=None).detach()
    membership = None
    while epoch < model.epoch:
        epoch += 1
        truth = []
        predicted_label = []
        for train_sample in batch(train_loader, model.batch_size):
            optimizer.zero_grad()
            v1 = torch.tensor(train_sample['data']).double().to(model.device)
            labels = torch.tensor(train_sample['label']).double().to(model.device)
            v2 = torch.tensor(train_sample['data2']).double().to(model.device)
            s = torch.tensor(train_sample['sensitive']).double().to(model.device)
            loss, membership = model(v1, v2, s, centroid, True, True,  mode=args.mode)
            l1 = loss[0]
            l2 = loss[1]
            l3 = loss[2]
            (l1 + l2 + l3).backward()
            optimizer.step()
            cur_iter += 1
            if epoch < model.epoch:
                centroid = model.compute_centroid(v1, v2, s, None, membership).detach()
            predicted = membership.mean(dim=0).argmax(dim=0)
            predicted_label = predicted_label + list(predicted.cpu().numpy())
            truth = truth + list(labels.cpu().numpy())
        if epoch % 10 == 0:
            average_ratio = torch.mean(s, dim=0).cpu().reshape(-1, 1)
            mem = F.one_hot(membership.mean(dim=0).argmax(dim=0), num_classes=model.k).T.double().cpu()
            cluster_ratio = torch.matmul(mem, s.cpu()).T / (torch.sum(mem.T, 0) + 1)
            score = (cluster_ratio - average_ratio).norm(2)
            nmi = normalized_mutual_info_score(np.array(truth), np.array(predicted_label))
            ari = adjusted_rand_score(np.array(truth), np.array(predicted_label))
            model.train()
            print('Epoch [{}/{}], time: {:.4f}, KL Loss: {:.4f}, Contr Loss: {:.4f}, Fair Loss: {:.4f},'
                  ' nmi:{:.4f}, ari:{:.4f}, Fairness score: {:.4f}, total loss: {:.4f}'.format(
                  epoch, model.epoch, time.time() - start_time, l1, l2, l3, nmi, ari, score, sum(loss)))
            start_time = time.time()
    torch.save(model.state_dict(), 'model_{}.ckpt'.format(args.hidden))
    print('Final Result: nmi:{:.4f}, ari:{:.4f}, Fairness score: {:.4f}'.format(nmi, ari, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fair-MVC algorithm')
    parser.add_argument('-d', dest='data', type=str, default='credit', help='which dataset is used for demo')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='The initial learning rate')
    parser.add_argument('-e', dest='epoch', type=int, default=1000, help='the total epoch for training')
    parser.add_argument('-hid', dest='hidden', type=int, default=50, help='the size of the hidden representation')
    parser.add_argument('-f', dest='fold', type=int, default=0, help='the index of 5 folds cross validation (0-4)')
    parser.add_argument('-alpha', dest='alpha', type=float, default=2, help='the value of alpha')
    parser.add_argument('-beta', dest='beta', type=float, default=0.01, help='the value of beta')
    parser.add_argument('-gamma', dest='gamma', type=float, default=10, help='the value of gamma')
    parser.add_argument('-weight_decay', dest='weight_decay', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('-miss', dest='miss',  action='store_true', help='missing feature scenario')
    parser.add_argument('-purturbed', dest='purturbed',  action='store_true', help='purturbed feature scenario')
    parser.add_argument('-mode', dest='mode',  type=str, default='contra', help='contra or noncontra')
    args = parser.parse_args()
    for i in range(1):
        args.index = i
        print('Round {}:'.format(i+1))
        main(args)
        print('\n\n')
