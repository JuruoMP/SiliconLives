# coding: utf-8

import math
import numpy as np
import scipy


class LogisticRegression(object):
    DISPLAY = 10
    train_epoch = 10000
    alpha = 0.001
    epsilon = 1e-12

    def __init__(self, data, label):
        """
        Pass data to Logistic Regression and train it.
        :param data: n_sample * n_dim
        :param label: n_sample, {0, 1}
        """
        self.data = data
        self.label = label
        self.w = np.random.randn(data.shape[1] + 1)
        self.train()

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self):
        one = np.ones([self.data.shape[0], 1])
        data = np.concatenate((self.data, one), axis=1)
        for p_it in range(self.train_epoch):
            h = self.predict_with_padding(data)
            if self.DISPLAY > 0 and p_it % (self.train_epoch / self.DISPLAY) == 0:
                loss = 0
                for i in range(data.shape[0]):
                    sample_data = data[i]
                    sample_h = self.predict_with_padding(sample_data)
                    sample_label = self.label[i]
                    loss += sample_label * math.log2(sample_h + self.epsilon) + \
                            (1.0 - sample_label) * math.log2(1.0 - sample_h + self.epsilon)
                loss /= data.shape[0]
                print('loss = %.4f' % loss)
            error = h - self.label
            gradient = np.dot(error, data)
            self.w -= self.alpha * gradient

    def predict(self, sample):
        one = np.ones([sample.shape[0], 1])
        sample = np.concatenate((sample, one), axis=1)
        ret = self.predict_with_padding(sample)
        ret[ret > 0.5] = 1
        ret[ret <= 0.5] = 0
        return ret

    def predict_with_padding(self, sample):
        z = np.dot(self.w, sample.T)
        h = self.sigmoid(z)
        return h


if __name__ == '__main__':
    sample_label_data = scipy.io.loadmat('data.mat')['data']
    sample_data = sample_label_data[:, :-1]
    label_data = sample_label_data[:, -1]
    lr = LogisticRegression(sample_data, label_data)
    correct, cnt = 0, 0
    for sample, label in zip(sample_data, label_data):
        ret = lr.predict(sample.reshape(1, -1))
        if ret == label:
            correct += 1
        cnt += 1
    print('lr acc = %.4f' % (correct / cnt))
