# coding: utf-8

import sys
from sys import version
import numpy as np
import mlpy
import scipy.io


if __name__ == '__main__':
    sample_label_data = scipy.io.loadmat('data.mat')['data']
    sample_data = sample_label_data[:, :-1]
    label_data = sample_label_data[:, -1] * 2 - 3
    kfdac = mlpy.KFDAC(lmb=0.01, kernel=mlpy.KernelGaussian(sigma=1.0))
    kfdac.learn(sample_data, label_data)
    correct, cnt = 0, 0
    for sample, label in zip(sample_data, label_data):
        ret = kfdac.pred(sample)
        if ret == label:
            correct += 1
        cnt += 1
    print('%.4f' % (correct / cnt))
