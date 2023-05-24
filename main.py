from data_loader import *
import time
from module import MvClustering
import torch
from sklearn.metrics import normalized_mutual_info_score
import argparse
import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')


def main(args):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu>=0 else 'cpu')
    args.device = device
    if args.data == 'credit':
        dataset = load_uci_credit(use_mask=args.miss, index=args.index, missing_ratio=0, purturbed=args.purturbed)
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
        batch_size = 5000
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
        raise(NameError('{} data set is not support yet'.format(args.data)))
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
    print('alpha = {}, beta = {}, gamma = {}'.format(args.alpha, args.beta, args.gamma))
    nmi, balance = train_model(model, dataset, args)
    return nmi, balance


def batch(iterable, n=1):
    l = len(iterable)
    # iterable.shuffle()
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def balance_score(y_hat, r):
    return (torch.mm(y_hat, r).min(dim=1)[0]/torch.sum(y_hat, dim=1)).min()


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
    temp_alpha = args.alpha
    while epoch < model.epoch:
        epoch += 1
        truth = []
        predicted_label = []
        r = []
        if epoch < 100:
            model.alpha = 0
        elif epoch % 2 == 0:
            model.alpha = temp_alpha
        else:
            model.alpha = 0
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
            r.append(s)
            predicted_label = predicted_label + list(predicted.cpu().numpy())
            truth = truth + list(labels.cpu().numpy())
        if epoch % 50 == 0:
            r = torch.stack(r).squeeze(dim=0)
            if len(r.shape) == 3:
                r = r.reshape(-1, r.shape[2])
            mem = F.one_hot(torch.LongTensor(predicted_label).to(model.device), num_classes=model.k).T.double()
            balance = balance_score(mem, r)
            nmi = normalized_mutual_info_score(np.array(truth), np.array(predicted_label))
            model.train()
            print('Epoch [{}/{}], time: {:.4f}, Total Loss: {:.4f}'.format(epoch, model.epoch, time.time() - start_time,
                                                                     sum(loss)))
            start_time = time.time()
    torch.save(model.state_dict(), 'model_{}.ckpt'.format(args.hidden))
    print('Final Result: nmi:{:.4f}, Balance score: {:.4f}'.format(nmi,  balance))
    return nmi, balance.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FairMVC algorithm')
    parser.add_argument('-d', dest='data', type=str, default='credit', help='which dataset is used for demo')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='The initial learning rate')
    parser.add_argument('-e', dest='epoch', type=int, default=1000, help='the total epoch for training')
    parser.add_argument('-hid', dest='hidden', type=int, default=50, help='the size of the hidden representation')
    parser.add_argument('-f', dest='fold', type=int, default=0, help='the index of 5 folds cross validation (0-4)')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.01, help='the value of alpha')
    parser.add_argument('-beta', dest='beta', type=float, default=0.1, help='the value of beta')
    parser.add_argument('-gamma', dest='gamma', type=float, default=0.005, help='the value of gamma')
    parser.add_argument('-weight_decay', dest='weight_decay', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('-miss', dest='miss',  action='store_true', help='missing feature scenario')
    parser.add_argument('-purturbed', dest='purturbed',  action='store_true', help='purturbed feature scenario')
    parser.add_argument('-mode', dest='mode',  type=str, default='noncontra', help='contra or noncontra')
    args = parser.parse_args()
    seeds = [2, 5, 6, 7, 8]
    nmi_list = []
    balance_list = []
    for i in range(5):
        args.index = i
        print('Round {}:'.format(i+1))
        np.random.seed(seeds[i])
        torch.random.manual_seed(seeds[i])
        nmi, balance = main(args)
        nmi_list.append(nmi)
        balance_list.append(balance)
        print('\n\n')
    print('NMI: {}'.format(nmi_list))
    print('Balance: {}'.format(balance_list))
    print('AVG NMI : {}'.format(np.mean(nmi_list)))
    print('AVG Balance : {}'.format(np.mean(balance_list)))
