# coding: utf-8

import sys
import os
import numpy as np
import torch
from torch.autograd import Variable

# Construct a uninitialized matrix
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)
print(x.size())

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = x + y
print(result)
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

x.copy_(y)

print(x == y)

ones = torch.ones(5)

p = x.numpy()
x.add_(1)
q = torch.from_numpy(np.ones(1))
# p also add 1

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)

x = Variable(torch.ones(2, 2), requires_grad=True)
print('x = %s' % x)
y = 3 * x * x + 2
y = Variable(y, requires_grad=True)
print('y = %s' % y)
z = y.mean()
print(x, z)
z.backward()
print('y.grad = %s' % y.grad)
print('x.grad = %s' % x.grad)