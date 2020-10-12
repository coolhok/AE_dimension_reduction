# coding=utf-8
from data import feature_dset
import models
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import os

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
num_epoch = 5
z_dimension = 64

# 创建对象
AE = models.autoencoder(z_dimension).to(device)

train_data = feature_dset(feature_path="./data/feature.csv",
                          label_path="./data/label.csv")

dataloader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)


# 是单目标二分类交叉熵函数
criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(AE.parameters(), lr=0.0003)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (f, _) in enumerate(dataloader):
        num_f = f.size(0)
        f = f.view(num_f,  1, 16, 16).type(torch.FloatTensor).to(device)
        code, decode = AE(f)  # 将真实图片放入判别器中
        loss = criterion(decode, f)
        ae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        loss.backward()  # 将误差反向传播
        ae_optimizer.step()  # 更新参数
        if (i + 1) % 5 == 0:
            print('Epoch[{}/{}],ae_loss:{:.6f} '.format(
                epoch, num_epoch, loss.item()/batch_size,
            ))

# 保存模型
torch.save(AE.state_dict(), './AE.pth')
