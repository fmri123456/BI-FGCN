import numpy as np
import math
import torch
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn import preprocessing


# 阈值处理
def shold_S(S, beta):
    S[S < beta] = 0
    return S


# 计算同步矩阵S
def set_S(H):
    H = H.numpy()
    num = H.shape[0]
    row = H.shape[1]
    # 初始化同步矩阵S
    S = np.zeros((num, row, row))
    for k in range(num):
        for i in range(row):
            for j in range(row):
                # 主对角线元素为0
                # 分别取i，j行，来计算H_ik*H_jk
                a = H[k][i]
                b = H[k][j]
                c = sum(np.multiply(a, b))

                a_2 = math.sqrt(sum(np.multiply(a, a)))
                b_2 = math.sqrt(sum(np.multiply(b, b)))
                if a_2 == 0 or b_2 == 0:
                    S[k][i][j] = 0
                else:
                    S[k][i][j] = c / (a_2 * b_2)
    S = torch.FloatTensor(S)
    # 阈值处理
    S = shold_S(S, 0.8)
    return S


# 计算偏好矩阵P
def set_P(H):
    H = H.numpy()
    num = H.shape[0]
    row = H.shape[1]
    # 初始化同步矩阵P
    P = np.zeros((num, row, row))
    for k in range(num):
        for i in range(row):
            for j in range(row):
                # 分别取i，j行，来计算H_ik*H_jk
                c = np.vstack((H[k][i], H[k][j])).T
                # maxlag : {int, Iterable[int]}
                #         If an integer, computes the test for all lags up to maxlag. If an
                #         iterable, computes the tests only for the lags in maxlag.
                granger_result = grangercausalitytests(c, maxlag=[2])
                # F1 = granger_result[2][0]['ssr_ftest'][0]
                # F2 = granger_result[2][0]['ssr_chi2test'][0]
                # F3 = granger_result[2][0]['lrtest'][0]
                F4 = granger_result[2][0]['params_ftest'][0]
                P[k][i][j] = F4
    # 数据归一化(MinMaxScaler)
    for n in range(num):
        min_max_scaler = preprocessing.MinMaxScaler()
        P[n] = min_max_scaler.fit_transform(P[n])
    P = torch.FloatTensor(P)
    return P
