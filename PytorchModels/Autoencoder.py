# coding: utf-8

import math
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1156)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc1 = nn.Linear(12, 16)
        self.enc2 = nn.Linear(16, 32)
        self.enc3 = nn.Linear(32, 32)
        self.dec1 = nn.Linear(32, 32)
        self.dec2 = nn.Linear(32, 16)
        self.dec3 = nn.Linear(16, 12)

    def forward(self, x):
        x = F.sigmoid(self.enc1(x))
        x = F.sigmoid(self.enc2(x))
        x = F.sigmoid(self.enc3(x))
        x = F.sigmoid(self.dec1(x))
        x = F.sigmoid(self.dec2(x))
        x = F.sigmoid(self.dec3(x))
        return x

    def inference(self, x):
        x = F.sigmoid(self.enc1(x))
        x = F.sigmoid(self.enc2(x))
        x = F.sigmoid(self.enc3(x))
        return x


autoencoder = Autoencoder()

dataset = []
for _ in range(10000):
    # for _ in range(5):
    data = []
    for _ in range(12):
        data.append(random.random())
    dataset.append(data)
dataset = np.array(dataset, dtype=np.float32)


def get_batch(dataset, batch_size):
    no = 0
    while True:
        if (no + 1) * batch_size < len(dataset):
            batch = dataset[no * batch_size: (no + 1) * batch_size]
            no += 1
        else:
            batch = dataset[no * batch_size:]
            no = 0
        yield batch


loss_function = nn.MSELoss()
optimizer = optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9)

epoch_loss = 10000
while epoch_loss > 0.01:

    epoch_loss = 0.0
    bach_generator = get_batch(dataset, 1000)
    batch = next(bach_generator)

    optimizer.zero_grad()
    for data in batch:
        data = autograd.Variable(torch.from_numpy(data))
        data = data.unsqueeze(0)
        # print('data = %s' % data)
        data_ = autoencoder(data)
        # print('data_ = %s' % data_)
        loss = loss_function(data_, data)  # 先写计算结果，再写target
        epoch_loss += loss
    epoch_loss.backward()
    optimizer.step()

    print('epoch loss: %s' % epoch_loss)
