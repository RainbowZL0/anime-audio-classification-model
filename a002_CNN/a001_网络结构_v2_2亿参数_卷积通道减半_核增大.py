import torch
import torch.nn as nn


class CNNv2(torch.nn.Module):
    def __init__(self):
        super(CNNv2, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            # 第一层卷积 常常要控制kernel_size stride padding三个参数 使得卷积不改变原来输入的张量size
            nn.Conv2d(in_channels=2,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),  # 意思是四周加一圈0。(a, b)代表上下都加a层，左右都加b层
                      bias=True
                      ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),  # 在本代码中，MaxPool使张量的横纵方向大小都-1
                         stride=(1, 1)),

            # 第二层卷积
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(5, 5),
                      stride=(1, 1),
                      padding=(2, 6),
                      dilation=(1, 3),
                      bias=True
                      ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(1, 1)),

            # 第三层卷积
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(5, 7),
                      stride=(1, 1),
                      padding=(4, 15),
                      dilation=(2, 5),
                      bias=True
                      ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(1, 1)
                         )
        )

        self.linear_layer = nn.Sequential(
            # 第一层全连接 转为256神经元
            nn.Linear(64 * 211 * 37, 256),  # 这里需要根据dim_t和n_mfcc修改。在保持卷积层参数不变的情况下，是原始输入大小-3
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # 第二层全连接
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # 第三层全连接
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # 第四层全连接
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            # 最后一层输出 由于使用了交叉熵损失，所以不用激活函数
            nn.Linear(64, 2)
        )

    def forward(self, x):
        predict = self.conv_layer(x)
        predict = predict.view(predict.size(0), -1)
        predict = self.linear_layer(predict)
        return predict
