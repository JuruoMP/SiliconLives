# coding: utf-8

import math
import numpy as np
import scipy.io
from sklearn import svm


if __name__ == '__main__':
    print('Loading data...')
    sample_label_data = scipy.io.loadmat('data.mat')['data']
    sample_data = sample_label_data[:, :-1]
    label_data = sample_label_data[:, -1] - 1
    clf = svm.SVC(kernel='rbf')
    print('Training...')
    clf.fit(sample_data, label_data)
    print('Evaluating...')
    correct, cnt = 0, 0
    for sample, label in zip(sample_data, label_data):
        ret = clf.predict(sample.reshape(1, -1))
        if ret == label:
            correct += 1
        cnt += 1
    print('svm acc = %.4f' % (correct / cnt))
