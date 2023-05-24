import itertools
import pandas as pd
import os
import numpy as np
import scipy.io as sio
import random
from sklearn import preprocessing
import math


class load_xrmb():
    def __init__(self, num_class, number=75000):
        num = int(number/ num_class)
        data = sio.loadmat('./data/XRMBf2KALDI_window7_single2.mat')
        index_list = list(map(lambda x: np.where(data['testLabel'] == x)[0][:num], range(num_class)))
        indices = np.array(list(itertools.chain.from_iterable(index_list)))
        shuffle = np.random.permutation(len(indices))
        indices = indices[shuffle]
        self.data2 = data['XTe2'][indices]
        self.k = num_class
        self.label = np.array(data['testLabel'][indices]).reshape(-1, )
        data2 = sio.loadmat('./data/XRMBf2KALDI_window7_single1.mat')
        self.data = data2['XTe1'][indices]
        # self.data = np.concatenate([self.data_1, self.data2], axis=1)
        self.s = np.zeros((number, 1))
        # self.data = self.data / np.max(self.data, axis=0)
        # self.data2 = self.data2 / np.max(self.data2, axis=0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'sensitive': self.s[idx], 'label': self.label[idx]}

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.label = self.label[idx]


class load_uci_credit():
    def __init__(self, missing_ratio=0.1, index=0, use_mask=False, purturbed=False):
        data = pd.read_csv('./dataset/UCI_Credit_Card.csv')
        # set the sensitive feature to be either 0 or 1
        sensitive_feature = data['SEX'].values - 1
        label = data['PAY_0'].values
        # filter out the samples from some small clusters
        selected_sample = label < 3
        label = label[selected_sample]
        sensitive_feature = sensitive_feature[selected_sample]
        header = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
                  'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                  'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default.payment.next.month']
        # normalize the data along each feature
        scaler = preprocessing.StandardScaler().fit(data[header].values)
        data = scaler.transform(data[header].values)[selected_sample]
        self.k = np.unique(label).shape[0]
        # select 1000 samples from each cluster
        indices = []
        for j in np.unique(label):
            idx = np.where(label == j)[0]
            indices = indices + list(idx)[: 1000]
        self.data = self.sigmoid(data[indices]-1)
        self.data2 = np.tanh(data[indices]-0.1)
        self.s = np.array([sensitive_feature[indices], 1-sensitive_feature[indices]]).T
        # print(self.s.sum(axis=0))
        self.label = label[indices]
        if not os.path.exists("./dataset/UCI_Credit_Card_mask_{}.npy".format(missing_ratio)):
        # if not os.path.exists("./dataset/UCI_Credit_Card_mask.npy"):
            self.mask = []
            self.mask2 = []
            for i in range(5):
                self.mask.append(np.random.binomial(1, missing_ratio, self.data.shape))
                self.mask2.append(np.random.binomial(1, missing_ratio, self.data.shape))
            np.save("./dataset/UCI_Credit_Card_mask_{}.npy".format(missing_ratio), [self.mask, self.mask2])
            self.mask, self.mask2 = self.mask[index], self.mask2[index]
        else:
            mask, mask2 = np.load("./dataset/UCI_Credit_Card_mask_{}.npy".format(missing_ratio))
            self.mask, self.mask2 = mask[index], mask2[index]
        if not os.path.exists("./dataset/UCI_Credit_Card_noise.npy"):
            noise_1 = []
            noise_2 = []
            for i in range(5):
                noise_1.append(np.random.normal(0, 1, size=self.data.shape))
                noise_2.append(np.random.normal(0, 1, size=self.data2.shape))
            np.save("./dataset/UCI_Credit_Card_noise.npy", [noise_1, noise_2])
            noise_1, noise_2 = noise_1[index], noise_2[index]
        else:
            noise_1, noise_2 = np.load("./dataset/UCI_Credit_Card_noise.npy")
            noise_1, noise_2 = noise_1[index], noise_2[index]
        if use_mask:
            self.mask = self.mask.astype(bool)
            self.mask2 = self.mask2.astype(bool)
            self.data[self.mask] = 0
            self.data2[self.mask2] = 0
        if purturbed:
            self.data[self.mask] = self.data[self.mask] + noise_1[self.mask]
            self.data2[self.mask2] = self.data2[self.mask2] + noise_2[self.mask2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'sensitive': self.s[idx], 'label': self.label[idx]}

    def sigmoid(self, data):
        return 1/(1+np.exp(-data))

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.s = self.s[idx]
        self.label = self.label[idx]


class load_zafar():
    """
      genZafarData(n = 10000; d = pi/4)
    Generate synthetic data from Zafar et al., 2017 Fairness Constraints: Mechanisms for Fair Classification.
    # Arguments
    - `n=10000` : number of samples
    - `d=pi/4` : discrimination factor
    # Returns
    - `X` : DataFrame containing features and protected attribute z {"A", "B"} where
            z="B" is the protected group.
    - `y` : Binary Target variable {-1, 1}
    """
    def __init__(self, n=10000, d=math.pi/4, idx=1, missing_ratio=0, use_mask=False, purturbed=False):
        if os.path.exists('./dataset/zafar_{}.mat'.format(idx)):
            data = sio.loadmat('./dataset/zafar_{}.mat'.format(idx))
        else:
            from scipy.stats import multivariate_normal
            from numpy import linalg as LA
            mu1, sigma1 = [1.2, 0.2],  [[5., 3.], [1., 5.]]
            mu2, sigma2 = [-0.2, -0.2], [[5., 1.],  [1., 3.]]
            npos = np.int64(math.floor(n/2))
            X1, y1 = self.genGaussian(mu1, sigma1,  1, npos)     # positive class
            X2, y2 = self.genGaussian(mu2, sigma2, -1, n - npos) # negative class
            perm = np.random.permutation(np.arange(n))
            X = np.concatenate([X1, X2], axis=0)[perm, :]
            y = np.concatenate([y1, y2], axis=0)[perm]
            rotation_mult = [[[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]] for _ in range(100)]
            rotation_mult = np.array(rotation_mult).reshape(-1, 200)
            X_aux = np.dot(X, rotation_mult.T)/(LA.norm(X, axis=1, keepdims=True) * LA.norm(rotation_mult, axis=1, keepdims=True).T)
            """ Generate the sensitive feature here """
            x_control = [] # this array holds the sensitive feature value
            for i in range(n):
                x = X_aux[i, :]
                # probability for each cluster that the point belongs to it
                p1 = multivariate_normal.pdf(x, mean=mu1, cov=sigma1)
                p2 = multivariate_normal.pdf(x, mean=mu2, cov=sigma2)
                # normalize the probabilities from 0 to 1
                s = p1 + p2
                p1 = p1/s # p2 = p2/s
                if np.random.uniform(0, 1) < p1:
                    x_control.append(0) # majority class
                else:
                    x_control.append(1)# protected class
            data = {'X': X, 's': np.array(x_control), 'y': y}
            sio.savemat('./dataset/zafar_{}.mat'.format(idx), data)
        self.data = np.tanh(data['X'])[:n, :]
        self.data2 = self.sigmoid(data['X'])[:n, :]
        # self.s = np.array(data['s']).T[:n]
        data['s'] = data['s'].reshape(-1,)
        self.s = np.array([data['s'], 1-data['s']]).T[:n]
        # print(self.s.sum(axis=0))
        self.label = data['y'].reshape(-1, )[:n]
        self.k = np.unique(data['y']).shape[0]
        index = idx
        if not os.path.exists("./dataset/zafar_mask_{}.npy".format(missing_ratio)):
            self.mask = []
            self.mask2 = []
            for i in range(5):
                self.mask.append(np.random.binomial(1, missing_ratio, self.data.shape))
                self.mask2.append(np.random.binomial(1, missing_ratio, self.data.shape))
            np.save("./dataset/zafar_mask_{}.npy".format(missing_ratio), [self.mask, self.mask2])
            self.mask, self.mask2 = self.mask[index], self.mask2[index]
        else:
            mask, mask2 = np.load("./dataset/zafar_mask_{}.npy".format(missing_ratio))
            self.mask, self.mask2 = mask[index], mask2[index]
        if not os.path.exists("./dataset/zafar_noise.npy"):
            noise_1 = []
            noise_2 = []
            for i in range(5):
                noise_1.append(np.random.normal(0, 1, size=self.data.shape))
                noise_2.append(np.random.normal(0, 1, size=self.data2.shape))
            np.save("./dataset/zafar_noise.npy", [noise_1, noise_2])
            noise_1, noise_2 = noise_1[index], noise_2[index]
        else:
            noise_1, noise_2 = np.load("./dataset/zafar_noise.npy")
            noise_1, noise_2 = noise_1[index], noise_2[index]
        if use_mask:
            self.mask = self.mask.astype(bool)
            self.mask2 = self.mask2.astype(bool)
            self.data[self.mask] = 0
            self.data2[self.mask2] = 0
        if purturbed:
            self.data[self.mask] = self.data[self.mask] + noise_1[self.mask]
            self.data2[self.mask2] = self.data2[self.mask2] + noise_2[self.mask2]


    def genGaussian(self, mean_in, cov_in, class_label, n):
        # nv = Distributions.MvNormal(mean_in, cov_in)
        X = np.random.multivariate_normal(mean_in, cov_in, (n, 100)).reshape(-1, 200)
        y = np.ones(n) * class_label
        return X, y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'sensitive': self.s[idx], 'label': self.label[idx]}

    def relu(self, data):
        idx = data < 0
        data[idx] = 0
        return data

    def sigmoid(self, data):
        return 1/(1+np.exp(-data))

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.s = self.s[idx]
        self.label = self.label[idx]


class load_bank():
    def __init__(self, missing_ratio=0.1, add_noise=True, index=0):
        raw_data = pd.read_csv('./dataset/bank.csv').values
        header_cat_idx = [1, 3, 5, 6]
        header_num_idx = [0, 10, 11, 13, 15, 16, 17, 18]
        label_idx = 20
        sensitive_idx = 2
        processed_data = []
        for i in header_num_idx:
            processed_data.append(list(map(float, raw_data[:, i])))
        for i in header_cat_idx:
            processed_data.append(self.process_categorical(raw_data, i))
        processed_data = np.array(processed_data).T
        sensitive_feature = np.array(self.process_categorical(raw_data, sensitive_idx)).T
        label = np.array(self.process_categorical(raw_data, label_idx))
        self.k = np.unique(label).shape[0]
        # select 1000 samples from each cluster
        indices = []
        for j in np.unique(label):
            idx = np.where(label == j)[0]
            indices = indices + list(idx)[: 2500]
        processed_data = processed_data[indices] / np.max(processed_data[indices], axis=0)
        self.data = self.sigmoid(processed_data)
        self.data2 = self.relu(processed_data)
        self.s = sensitive_feature[indices]
        self.s = np.array([self.s, 1-self.s]).T
        # print(self.s.sum(axis=0))
        self.label = label[indices]
        # import collections
        # counter = collections.Counter(self.label)
        # print(counter)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'sensitive': self.s[idx], 'label': self.label[idx]}

    def relu(self, data):
        idx = data < 0
        data[idx] = 0
        return data

    def sigmoid(self, data):
        return 1/(1+np.exp(-data))

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.s = self.s[idx]
        self.label = self.label[idx]

    def process_categorical(self, data, idx):
        dictionary = dict()
        count = 0
        categorical = []
        for item in data[:, idx]:
            if item not in dictionary:
                dictionary[item] = count
                count += 1
            categorical.append(dictionary[item])
        return categorical


    def one_hot(self, data):
        num_unique = np.unique(data).shape[0]
        data = np.array(data)
        new_data = np.zeros((data.shape[0], num_unique))
        for i in np.unique(data):
            idx = np.argwhere(data == i)
            new_data[idx, i] = 1
        # remove some columns if it is too sparse
        item_count_ratio = new_data.sum(axis=0) / new_data.sum()
        idx = item_count_ratio > 0.04
        return new_data[:, idx]


class load_mnist():
    def __init__(self, num, noise_rate=0.2, missing_ratio=0.1, index=0, use_mask=False):
        self.num_class = 10
        if not os.path.exists('./data/mnist/noisy_mnist_train_2v_{}.mat'.format(noise_rate)):
            import tensorflow as tf
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
            view_1 = []
            view_2 = []
            import cv2
            import imutils
            for i in range(num):
                angle = random.randint(-45, 45)
                img = imutils.rotate(x_train[i], angle).reshape(28 * 28, )
                view_1.append(np.array(img))
                blur_img = cv2.blur(x_train[i].reshape((28, 28, 1)), (4, 4)).reshape(28 * 28, )
                view_2.append(np.array(blur_img))
            sio.savemat('./data/mnist/noisy_mnist_train_2v_{}.mat'.format(noise_rate),
                        {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'label': np.array(y_train)})
            self.data = view_1[:num]
            self.data2 = view_2[:num]
            self.s = np.array(range(y_train[:num].shape[0]))
            self.label = y_train[:num]
            self.k = 10
        else:
            data = sio.loadmat('./data/mnist/noisy_mnist_train_2v_{}.mat'.format(noise_rate))
            self.data = data['view_1'][:num]
            self.data2 = data['view_2'][:num]
            self.s = np.zeros((num, 1))
            self.label = data['label'][0][:num]
            self.k = 10
            from collections import Counter
            # print(Counter(self.label))
        if not os.path.exists("./data/mnist/noisy_mnist_train_2v_mask_{}.npy".format(missing_ratio)):
            self.mask = []
            self.mask2 = []
            for i in range(5):
                self.mask.append(np.random.binomial(1, missing_ratio, self.data.shape))
                self.mask2.append(np.random.binomial(1, missing_ratio, self.data.shape))
            np.save("./data/mnist/noisy_mnist_train_2v_mask_{}.npy".format(missing_ratio), [self.mask, self.mask2])
            self.mask, self.mask2 = self.mask[index], self.mask2[index]
        else:
            mask, mask2 = np.load("./data/mnist/noisy_mnist_train_2v_mask_{}.npy".format(missing_ratio))
            self.mask, self.mask2 = mask[index], mask2[index]
        if use_mask:
            self.mask = self.mask.astype(bool)
            self.mask2 = self.mask2.astype(bool)
            self.data[self.mask] = 0
            self.data2[self.mask2] = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'data2': self.data2[idx], 'sensitive': self.s[idx], 'label': self.label[idx]}

    def shuffle(self):
        idx = np.random.permutation(range(len(self.data)))
        self.data = self.data[idx]
        self.data2 = self.data2[idx]
        self.s = self.s[idx]
        self.label = self.label[idx]
