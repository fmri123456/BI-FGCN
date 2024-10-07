import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn


# 同步模糊卷积
class Synchronized_fuzzy_Convolution_layer(nn.Module):
    def __init__(self, use_bias=False):
        super(Synchronized_fuzzy_Convolution_layer, self).__init__()
        self.use_bias = use_bias
        self.theta1 = Parameter(torch.FloatTensor(70, 70), requires_grad=True)
        self.reset_parameters()

    def save_weight(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta1.size(1))
        self.theta1.data.uniform_(-stdv, stdv)


    def forward(self, S, F, H, A):
        num = H.size(0)
        row = H.size(1)
        D = torch.zeros([num, row, row])
        D1 = torch.zeros([num, row])
        for n in range(num):
            for i in range(row):
                D1[n][i] = torch.sum(A[n][i])
        D_12 = D1 ** (-0.5)  # 负1/2次幂
        for j in range(num):
            D[j] = torch.diag(D_12[j])  # D的负1/2次幂
        SF = S * F  # S,F的哈达玛积
        H_s = torch.matmul(torch.bmm(torch.bmm(torch.bmm(D, SF), D), H), self.theta1)
        return torch.relu(H_s)


# 偏好模糊卷积
class Preference_fuzzy_Convolution_layer(nn.Module):
    def __init__(self, use_bias=False):
        super(Preference_fuzzy_Convolution_layer, self).__init__()
        self.use_bias = use_bias
        self.theta2 = Parameter(torch.FloatTensor(70, 70), requires_grad=True)
        self.reset_parameters()

    def save_weight(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta2.size(1))
        self.theta2.data.uniform_(-stdv, stdv)


    def forward(self, P, F, H, A):
        num = H.size(0)
        row = H.size(1)
        D = torch.zeros([num, row, row])
        D1 = torch.zeros([num, row])
        for n in range(num):
            for i in range(row):
                D1[n][i] = torch.sum(A[n][i])
        D_12 = D1 ** (-0.5)  # 负1/2次幂
        for j in range(num):
            D[j] = torch.diag(D_12[j])  # D的负1/2次幂
        PF = P * F  # S,F的哈达玛积
        H_1 = torch.matmul(torch.bmm(torch.bmm(torch.bmm(D, PF), D), H), self.theta2)
        return torch.relu(H_1)
