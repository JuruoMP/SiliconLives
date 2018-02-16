# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, x_dim, y_dim=1):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(x_dim, y_dim)

    def forward(self, x):
        y = self.fc(x)
        return y


xs = np.array([[1,2,3], [4,5,6], [7,8,9], [1,3,5], [2,4,6], [3,5,7], [4,6,8], [5,7,9], [1,5,9]])
ys = 3 * xs[:, 0] + 2 * xs[:, 1] + xs[:, 2]
assert len(xs) == len(ys)
MAX_EPOCH = 1000

linear_regression = LinearRegression(len(xs[0]))
xs = Variable(torch.Tensor(xs))
ys = Variable(torch.Tensor(ys))
if torch.cuda.is_available():
    linear_regression = linear_regression.cuda()
    xs = xs.cuda()
    ys = ys.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_regression.parameters(), lr=1e-2)
for _ in range(MAX_EPOCH):
    ys_ = linear_regression(xs)
    loss = criterion(ys_, ys)
    if _ % 100 == 0:
        print('loss = {}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = Variable(torch.Tensor([[1,2,3], [4,5,6]]))
y = linear_regression(x)
print(y)
