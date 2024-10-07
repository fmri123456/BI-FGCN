# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import torch.optim as optim
from models import FGCN
import utils_graph
import torch.utils.data as Data
import load_data
import numpy as np
import torch.nn as nn
print('cuda:', torch.cuda.is_available())
# 导入数据
feat_train, feat_test, P_train, P_test, label_train, label_test = load_data.load_data()
batch_size = 16

dataset = Data.TensorDataset(feat_train, P_train, label_train)
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

label_test = label_test.detach().numpy()
label_test = utils_graph.onehot_encode(label_test)
label_test = torch.LongTensor(np.where(label_test)[1])

model = FGCN()

LR = 1e-5
EPOCH = 10
max_acc = 0
loss_list = []
acc_list = []
out_data = torch.zeros(470, 2)
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-3)  # optimize all fgcn parameters
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (H, P, label) in enumerate(train_loader):
        model.train()
        print("第{}轮第{}批数据训练...".format(epoch + 1, step + 1))
        _, output = model.forward(H, P)
        label = label.detach().numpy()
        label = utils_graph.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        loss = loss_func(output, label)
        # print("loss:", loss)
        acc_val = utils_graph.accuracy(output, label)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()

        if epoch == EPOCH - 1:
            if output.shape[0] == batch_size:
                out_data[step * 16:(step + 1) * output.shape[0], :] = output

    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

        model.eval()
        _, output1 = model(feat_test, P_test)
        loss_val1 = loss_func(output1, label_test)
        acc_val1 = utils_graph.accuracy(output1, label_test)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "loss= {:.4f}".format(loss_val1.item()),
              "accuracy= {:.4f}".format(acc_val1.item()))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        loss_list.append(float(loss_val1.item()))
        acc_list.append(float(acc_val1.item()))
    if max_acc < acc_val1:
        max_acc = acc_val1
print("The model best accuracy:{:.4f}".format(max_acc))
# 保存模型参数
torch.save(model.state_dict(), 'FGCN.pth')



