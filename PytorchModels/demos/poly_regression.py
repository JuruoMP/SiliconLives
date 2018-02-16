# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class PolyRegression(nn.Module):
    def __init__(self, x_dim, y_dim=1, degree=2):
        super(PolyRegression, self).__init__()
        self.degree = degree
        self.fc = nn.Linear(x_dim * degree, y_dim)
        self.bias = nn.Linear(1, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = torch.cat([x ** i for i in range(1, self.degree + 1)], 2)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        y += self.bias(Variable(torch.Tensor([1])))
        return y


xs = np.array([[1,2,3], [4,5,6], [7,8,9], [1,3,5], [2,4,6], [3,5,7], [4,6,8], [5,7,9], [1,5,9], [2,5,8]])
xs = torch.Tensor(xs)
def f(x):
    Ws = torch.Tensor([1,2,3]).unsqueeze(1)
    b = torch.Tensor([-1])
    return x.mm(Ws) + b
# ys = f(xs)
ys = xs[:, 0] ** 3 + xs[:, 1] ** 2 + xs[:, 2]
assert len(xs) == len(ys)
MAX_EPOCH = 10000

model = PolyRegression(len(xs[0]), degree=3)
xs = Variable(xs)
ys = Variable(ys)
if torch.cuda.is_available():
    model = model.cuda()
    xs = xs.cuda()
    ys = ys.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for _ in range(MAX_EPOCH):
    ys_ = model(xs)
    loss = criterion(ys_, ys)
    if _ % 1000 == 0:
        print('loss = {}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = Variable(torch.Tensor([[1,2,3], [4,5,6]]))
y = model(x)
print(y)
