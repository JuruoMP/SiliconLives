# coding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init()__