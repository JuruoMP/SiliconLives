# coding： utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.functional as F

class SeqFCNet(nn.Module):
    def __init__(self, x_dim, y_dim=1, hidden_dim=16):
        super(SeqFCNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), 
            # nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(p=0.4), 
            nn.Sigmoid())
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            # nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(p=0.4), 
            nn.Sigmoid())
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, y_dim), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


xs = np.array([[1,2,3], [4,5,6], [7,8,9], [1,3,5], [2,4,6], [3,5,7], [4,6,8], [5,7,9], [1,5,9]])
ys = (3 * xs[:, 0] - 2 * xs[:, 1] ** 2 + xs[:, 2] ** 3 > 200).astype(float)
assert len(xs) == len(ys)
MAX_EPOCH = 5000

model = SeqFCNet(len(xs[0]), hidden_dim=16)
xs = Variable(torch.Tensor(xs))
ys = Variable(torch.Tensor(ys))
if torch.cuda.is_available():
    model = model.cuda()
    xs = xs.cuda()
    ys = ys.cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
for _ in range(MAX_EPOCH):
    ys_ = model(xs)
    loss = criterion(ys_, ys)
    if _ % 500 == 0:
        print('loss = {}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = Variable(torch.Tensor([[1,2,3], [4,5,6]]))
y = model(x)
print(y)
