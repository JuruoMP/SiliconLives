# coding: utf-8

import math
import random
import numpy as np
import scipy.io


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def mat_dot(mats):
    """
    多矩阵乘法
    :param mats:矩阵列表
    :return: 矩阵乘积
    """
    ret = mats[0]
    for mat in mats[1:]:
        ret = np.dot(ret, mat)
    return ret


class MLP(object):

    batch_size = 200
    regular_lambda = 0.001
    alpha = 0.0001
    max_epoch = 500
    momentum = 0.9
    display = 10

    def __init__(self, data, label, hidden_dim, num_classes):
        """
        传入样本，并训练模型
        :param data: 数据样本，n_sample*n_dim
        :param label: 标签n_sample
        :param hidden_dims: list, 隐藏层神经元数量
        """
        self.data = data
        self.label = label
        self.n_dim = self.data.shape[1]
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.W1 = self.W2 = None
        self.b1 = self.b2 = None
        self.dW1 = self.dW2 = self.db1 = self.db2 = None
        self.build_model()
        self.train()

    def build_model(self):
        self.W1 = np.random.randn(self.n_dim, self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.num_classes)
        self.b1 = np.random.randn(self.hidden_dim)
        self.b2 = np.random.randn(self.num_classes)

    def forward(self, x, y=None, mode='Eval'):
        # forward
        z1 = np.dot(x, self.W1) + self.b1  # z1 = W1 * x + b1
        a1 = sigmoid(z1)  # a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2)  # z2 = W2 * a1
        probs = np.exp(z2)  # probs = softmax(z2)
        probs /= np.sum(probs, axis=1, keepdims=True)
        if mode == 'Eval':
            return probs
        elif mode == 'Train':
            assert y is not None
            log_probs = [-np.log(probs[i, int(y[i])]) for i in range(x.shape[0])]
            loss = np.sum(log_probs)
            # backward
            for i in range(x.shape[0]):
                probs[i, int(y[i])] -= 1
            delta3 = probs
            dW2 = np.dot(a1.T, delta3)
            if self.db2 is None:
                self.db2 = np.sum(delta3, axis=0)
            else:
                self.db2 = self.momentum * self.db2 + (1 - self.momentum) * np.sum(delta3, axis=0)
            delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(x.T, delta2)
            if self.db1 is None:
                self.db1 = np.sum(delta2, axis=0)
            else:
                self.db1 = self.momentum * self.db1 + (1 - self.momentum) * np.sum(delta2, axis=0)
            dW2 += self.regular_lambda * self.W2
            dW1 += self.regular_lambda * self.W1
            if self.dW2 is None:
                self.dW2 = dW2
            else:
                self.dW2 = self.momentum * self.dW2 + (1 - self.momentum) * np.dot(a1.T, delta3)
            if self.dW1 is None:
                self.dW1 = dW1
            else:
                self.dW1 = self.momentum * self.dW1 + (1 - self.momentum) * np.dot(x.T, delta2)
            return probs, loss, self.dW1, self.dW2, self.db1, self.db2

    def train(self):
        num_batches = math.ceil(self.data.shape[0] / self.batch_size)
        for epoch in range(self.max_epoch):
            loss = 0
            for i in range(num_batches):
                samples = self.data[self.batch_size * i: self.batch_size * (i + 1), ]
                labels = self.label[self.batch_size * i: self.batch_size * (i + 1), ]
                probs, batch_loss, dW1, dW2, db1, db2 = self.forward(samples, labels, mode='Train')
                loss += batch_loss
                self.W1 -= self.alpha * dW1
                self.W2 -= self.alpha * dW2
                self.b1 -= self.alpha * db1
                self.b2 -= self.alpha * db2
            if epoch and epoch % 100 == 0:
                self.alpha /= 10
                self.dW1 = self.dW2 = self.db1 = self.db2 = None
            if self.display and epoch % self.display == 0:
                print('epoch %d, loss = %.4f' % (epoch, loss))

    def predict(self, sample):
        sample = np.array(sample)
        probs = self.forward(sample, mode='Eval')
        ret = []
        for prob in probs:
            ret.append(1 if prob[0] < prob[1] else 0)
        return ret


if __name__ == '__main__':
    sample_label_data = scipy.io.loadmat('data.mat')['data']
    np.random.shuffle(sample_label_data)
    sample_data = sample_label_data[:, :-1]
    label_data = sample_label_data[:, -1] - 1
    mlp = MLP(sample_data, label_data, 50, 2)
    correct, cnt = 0, 0
    for sample, label in zip(sample_data, label_data):
        ret = mlp.predict(sample.reshape(1, -1))
        if ret == label:
            correct += 1
        cnt += 1
    print('fisher acc = %.4f' % (correct / cnt))
