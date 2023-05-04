import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            # 第一层卷积 常常要控制kernel_size stride padding三个参数 使得卷积不改变原来输入的张量size
            nn.Conv2d(in_channels=2,
                      out_channels=32,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),  # 意思是四周加一圈0。(a, b)代表上下都加a层，左右都加b层
                      bias=True
                      ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),  # 在本代码中，MaxPool使张量的横纵方向大小都-1
                         stride=(1, 1)),

            # 第二层卷积
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=True
                      ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(1, 1)),

            # 第三层卷积
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=True
                      ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(1, 1)
                         )
        )

        self.linear_layer = nn.Sequential(
            # 第一层全连接 转为256神经元
            nn.Linear(128 * 211 * 37, 256),  # 这里需要根据dim_t和n_mfcc修改。在保持卷积层参数不变的情况下，是原始输入大小-3
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # 第二层全连接
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # 第三层全连接
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            # 第四层全连接
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            # 最后一层输出 由于使用了交叉熵损失，所以不用激活函数
            nn.Linear(16, 2)
        )

    def forward(self, x):
        predict = self.conv_layer(x)
        predict = predict.view(predict.size(0), -1)
        predict = self.linear_layer(predict)
        return predict


# 和上面CNN一样，只是网络训练过程拆成了一层一层的分开的步骤，方便debug
class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        conv_kernel_size = (3, 3)
        conv_stride = (1, 1)
        conv_padding = (1, 1)

        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.fc1 = nn.Linear(999296, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.relu6 = nn.LeakyReLU()

        self.fc4 = nn.Linear(64, 16)
        self.bn7 = nn.BatchNorm1d(16)
        self.relu7 = nn.LeakyReLU()

        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Linear layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.fc4(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.fc5(x)
        return x
