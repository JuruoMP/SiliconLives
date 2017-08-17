# coding: utf-8
'''
多项式拟合
Polynomial Curve Fitting
'''

import math
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tensorflow as tf

DEGREE = 3
FUNCTION = lambda x: x * x#math.sin(x)

def get_poly_func(_degree):
    scope = {}
    scope['poly_func'] = None
    # poly_func(_input, _pv0, _pv1, ...)
    func_str = 'def poly_func(_input, '
    for i in range(_degree + 1):
        func_str += '_pv' + str(i) + ', '
    func_str = func_str[:-2] + '):'
    func_str += '''
    ret = 0
    parameters = sorted(dict(locals()).items(), key=lambda x: x[0])[1:]
    for i, (paramater_name, parameter) in enumerate(parameters):
        degree = %s - i
        ret += parameter * (_input ** degree)
    return ret
    ''' % (_degree)
    exec(func_str, scope)
    return scope['poly_func']
poly_func = get_poly_func(DEGREE)

xs = np.array(sorted(np.random.random(size=[20]) * 10))
ys = np.array([FUNCTION(_) for _ in xs])

# 最小二乘法 手工实现
def poly_fit_LS(_xs, _ys, _DEGREE):   # Least squares
    X = np.matrix([_xs ** (_DEGREE - i) for i in range(_DEGREE + 1)]).T
    print('X.shape = %s' % str(X.shape))
    Y = np.matrix(_ys).T
    print('Y.shape = %s' % str(Y.shape))
    A = (X.T * X).I  * X.T * Y
    return A
func_fit = poly_fit_LS(xs, ys, DEGREE)
print(func_fit)

# 梯度下降 TensorFlow实现
# TODO: not implemented

# 最小二乘法 numpy实现
func_fit = np.polyfit(xs, ys, DEGREE)
print('parameters = %s' % func_fit)
plt.plot(xs, poly_func(xs, *func_fit), 'r-', label='fit')
plt.plot(xs, ys, 'b-', label='original')
plt.show()

# scipy实现
popt, pcov = curve_fit(poly_func, xs, ys)
# 说明：函数不仅可以拟合多项式，也可以拟合各种复杂函数
print(popt, pcov)
plt.plot(xs, poly_func(xs, *popt), 'r-', label='fit')
plt.plot(xs, ys, 'b-', label='original')
plt.show()
