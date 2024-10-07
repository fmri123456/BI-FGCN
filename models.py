import layer
import torch.nn as nn
import torch.nn.functional as F
import fuzzy_matrix
import construct_fuzzygraph
import torch


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


