# coding: utf-8

import math
import numpy as np
import scipy.io
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

"""
使用Adaboost算法对数据点进行分类
Adaboost融合了200个深度为1的决策树，最终达到预测结果准确率93。8%
"""
if __name__ == '__main__':
    print('Loading data...')
    sample_label_data = scipy.io.loadmat('data.mat')['data']
    sample_data = sample_label_data[:, :-1]
    label_data = sample_label_data[:, -1] * 2 - 3
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),\
                             algorithm='SAMME', n_estimators=200)
    print('Training...')
    bdt.fit(sample_data, label_data)
    print('Evaluating...')
    correct, cnt = 0, 0
    for sample, label in zip(sample_data, label_data):
        ret = bdt.predict(sample.reshape(1, -1))
        if ret == label:
            correct += 1
        cnt += 1
    print('svm acc = %.4f' % (correct / cnt))
