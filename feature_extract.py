import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fuzzy_matrix
import construct_fuzzygraph
import layer
import load_data


class FGCN(nn.Module):
    def __init__(self):
        super(FGCN, self).__init__()
        # 同步模糊卷积
        self.conv1 = layer.Synchronized_fuzzy_Convolution_layer()
        # 偏好模糊卷积
        self.conv2 = layer.Preference_fuzzy_Convolution_layer()
        # 全连接层
        self.fc1 = nn.Linear(161 * 161, 161)
        self.fc2 = nn.Linear(161, 70 // 2)
        self.fc3 = nn.Linear(70 // 2, 2)

    def forward(self, H1, P1):
        # 构建模糊图
        F1, A = construct_fuzzygraph.fuzzygraph(H1)
        # F1 = torch.tensor(F1)
        # A = torch.tensor(A)
        # 构建同步模糊矩阵和偏好模糊矩阵
        S1 = fuzzy_matrix.set_S(H1)
        # S1 = torch.tensor(S1)
        print("S计算完成")
        fgcn1 = F.relu(self.conv1.forward(S1, F1, H1, A))
        print("同步卷积完成")
        fgcn2 = F.relu(self.conv2.forward(P1, F1, fgcn1, A))
        print("偏好卷积完成")
        # 更新模糊矩阵
        F2, _ = construct_fuzzygraph.fuzzygraph(fgcn2)
        gc2_rl = F2.reshape(-1, 161 * F2.shape[2])
        fc1 = F.relu(self.fc1(gc2_rl))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.softmax(self.fc3(fc2), dim=1)
        # print("fc3:", fc3)
        # fc4 = F.softmax(fc3, dim=1)
        # fc4 = torch.sigmoid(fc3)
        # fgcn2输出更新后的特征矩阵, fc3输出全连接层后的结果
        return fgcn2, fc3


def fre_statis(fc1_w, fc2_w, fc3_w, fc4):
    fc4_imp = np.zeros(2)
    fc3_imp = np.zeros(35)
    fc2_imp = np.zeros(161)
    fc1_imp = np.zeros(161 * 161)
    imp = torch.zeros(fc4.shape[0], 161)
    imp_idx = torch.zeros(fc4.shape[0], 161)
    edge_score = torch.zeros([fc4.shape[0], 161, 161])
    best_edge = torch.zeros([fc4.shape[0], 161, 161])
    z = 0
    for kk in range(fc4.shape[0]):
        print("num:", kk)
        temp = fc4[kk]
        if temp[1] > temp[0]:
            fc4_imp[0] = (math.exp(temp[0]) * math.exp(temp[1])) / (
                    (math.exp(temp[0]) + math.exp(temp[1])) * (math.exp(temp[0]) + math.exp(temp[1])))
            fc4_imp[1] = 0
            # 得到第四层2个神经元的值

            for i in range(0, 35):
                fc3_imp[i] = 0
                for j in range(0, 2):
                    fc3_imp[i] = fc3_imp[i] + fc3_w[j][i] * fc4_imp[j]
            # 得到第三层35个神经元的值

            for i in range(0, 161):
                fc2_imp[i] = 0
                for j in range(0, 35):
                    fc2_imp[i] = fc2_imp[i] + fc2_w[j][i] * fc3_imp[j]
            # 得到第二层161个神经元的值

            for i in range(0, 161 * 161):
                fc1_imp[i] = 0
                for j in range(0, 161):
                    fc1_imp[i] = fc1_imp[i] + fc1_w[j][i] * fc2_imp[j]

            H = np.zeros(161)
            for i in range(0, 161):
                for j in range(161 * i, 161 * (i + 1)):
                    H[i] = H[i] + fc1_imp[j]
        else:
            fc4_imp[1] = (math.exp(temp[0]) * math.exp(temp[1])) / (
                    (math.exp(temp[0]) + math.exp(temp[1])) * (math.exp(temp[0]) + math.exp(temp[1])))
            fc4_imp[0] = 0

            for i in range(0, 35):
                fc3_imp[i] = 0
                for j in range(0, 2):
                    fc3_imp[i] = fc3_imp[i] + fc3_w[j][i] * fc4_imp[j]

            for i in range(0, 161):
                fc2_imp[i] = 0
                for j in range(0, 35):
                    fc2_imp[i] = fc2_imp[i] + fc2_w[j][i] * fc3_imp[j]

            for i in range(0, 161 * 161):
                fc1_imp[i] = 0
                for j in range(0, 161):
                    fc1_imp[i] = fc1_imp[i] + fc1_w[j][i] * fc2_imp[j]

            H = np.zeros(161)
            for i in range(0, 161):
                for j in range(161 * i, 161 * (i + 1)):
                    H[i] = H[i] + fc1_imp[j]

        B = np.argsort(H)
        B = list(reversed(B))  # B中存储排序后的下标
        A = sorted(H, reverse=True)  # A中存储排序后的结果
        AA = torch.tensor(A)
        BB = torch.tensor(B)
        imp[z] = AA
        imp_idx[z] = BB
        edge_score[z] = sorted(fc1_imp, reverse=True)
        best_edge[z] = list(reversed(np.argsort(fc1_imp)))

        z += 1
        # print(imp.shape)
    # 提取每个人特征排序
    vertex_score = imp
    best_vertex = imp_idx

    return best_vertex, best_edge, vertex_score, edge_score


# 特征提取
feat_train, feat_test, P_train, P_test, label_train, label_test = load_data.load_data()
model = FGCN()
model.load_state_dict(torch.load('FGCN.pth'))
_, output = model(feat_test, P_test)
fc1_w = model.state_dict()['fc1.weight']
fc2_w = model.state_dict()['fc2.weight']
fc3_w = model.state_dict()['fc3.weight']
fc4 = output
best_vertex, best_edge, vertex_score, edge_score = fre_statis(fc1_w, fc2_w, fc3_w, fc4)

np.save('best_vertex.npy', best_vertex)
np.save('best_edge.npy', best_edge)
np.save('vertex_score.npy', vertex_score)
np.save('edge_score.npy', edge_score)
