import torch
import math
import numpy as np
import torch.nn.functional as Fun
# import cupy as np


# 权重矩阵(皮尔逊相关系数)
def set_weight(X):
    X = X.detach().numpy()
    # 节点个数
    row = X.shape[0]
    # 特征个数
    col = X.shape[1]
    # 初始化权重矩阵
    W = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                if j != i:
                    # Pearson
                    W[k][i][j] = np.corrcoef(X[k][i], X[k][j])[0, 1]
    W = weight_threshold(W)
    W = torch.FloatTensor(W)
    return W


# 权重矩阵阈值化(关联系数最小的20%元素置0)
def weight_threshold(W):
    row = W.shape[0]
    col = W.shape[1]
    result = np.zeros((row, col, col))
    for i in range(row):
        threshold = np.sort(np.abs(W[i].flatten()))[int(col * col * 0.2)]  # 阈值
        result[i] = W[i] * (np.abs(W[i]) >= threshold)
    return result


# 邻接矩阵
def set_A(W):
    W = W.detach().numpy()
    # 节点个数
    row = W.shape[0]
    # 特征个数
    col = W.shape[1]
    # 初始化邻接矩阵
    A = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                if W[k][i][j] != 0:
                    A[k][i][j] = 1
            A[k][i][i] = 1
    A = torch.FloatTensor(A)
    return A


# 欧氏距离矩阵dir
def set_dir(X):
    X = X.detach().numpy()
    row = X.shape[0]
    col = X.shape[1]
    # 初始化欧氏距离矩阵
    dis = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                dis[k][i][j] = np.linalg.norm(X[k][i] - X[k][j])
    dis = torch.FloatTensor(dis)
    return dis


# 计算间接关系矩阵Ind
def set_Ind(X):
    X = X.detach().numpy()
    row = X.shape[0]
    col = X.shape[1]
    # 初始化间接关系矩阵Ind
    Ind = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                # 分别取i，j行，来计算|C_ik-C_jk|
                a = X[k][i]
                b = X[k][j]
                c = b - a
                # 对c的所以元素取绝对值
                abs = np.maximum(c, -c)
                Ind[k][i][j] = sum(abs)
    Ind = torch.FloatTensor(Ind)
    return Ind


# 融合直接关系和间接关系
def norm_relation(Dir, Ind):
    temp = Dir + Ind
    # 数据归一化
    result = torch.nn.functional.normalize(temp, p=2, dim=0)
    return result


# 计算关联关系矩阵F
def set_F(result):
    sig = math.sqrt(1)
    row = result.size(0)
    col = result.size(1)
    # 初始化关联关系矩阵F
    F = torch.zeros([row, col, col])
    for k in range(row):
        for i in range(col):
            for j in range(col):
                F[k][i][j] = torch.exp(-(result[k][i][j] - 2) ** 2 / (2 * sig ** 2))
    return F

# 阈值处理
def shold_F(F, alpha):
    F[F < alpha] = 0
    return F

# =====构建模糊图======
def fuzzygraph(feat):
    # 直接关系
    Dir = set_dir(feat)
    # 权重（皮尔逊系数）
    w = set_weight(feat)
    # 邻接矩阵
    A = set_A(w)
    # 间接关系
    Ind = set_Ind(w)
    # 融合
    result = norm_relation(Dir, Ind)
    # 构建脑区-基因模糊图
    F = set_F(result)
    # 阈值处理
    F = shold_F(F, 0.7)
    return F, A
