"""
提升方向：
1. 提升特征提取方法。当前的方法是mfcc()后得到(n_mfcc, t)维度的矩阵，然后对时域取平均值，每个音频只得到40个特征，丢弃了大部分的时域序列信息。是否可以不取平均。
2. 模型优化。卷积神经网络+循环神经网络，对音色、语调的识别效果更好。
3. 训练集提纯。
"""

import os
import librosa
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile

# 训练集目录，分为tom的，不是tom的。文件夹内应该有100个以上的wav文件，每个3s左右
tom_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\训练集\\钉宫语音_随机切分_最大4.9s'
not_tom_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\训练集\\其他语音_随机切分_最大4.9s'

# 待判断是不是tom语音的目录
test_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\测试集'


# 其他的代码和原始代码相同

# 提取MFCC特征 mel频率
def extract_features(file_path):
    wav_file, sample_rate = torchaudio.load(file_path)
    wav_file = wav_file.to('cuda')
    # n_mfcc是特征数量。40已经算比较大了，更大可能过拟合。
    # n_fft 短时傅里叶变换的段落长度，单位是采样点数。如果更注重短时间内的动态特征，应该调大一点。如果更注重频率特征，应该降至1024
    # hop_length 短时傅里叶变换的窗口滑动长度，单位是采样点数。一般设为n_fft的一半或更小
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={'n_fft': 2048, 'hop_length': 1024}
    )
    mfcc_matrix = mfcc_transform(wav_file)  # 返回张量(2, n_mfcc, t)
    mfcc_matrix = torch.mean(mfcc_matrix, dim=0)
    mfcc_matrix = torch.mean(mfcc_matrix, dim=1)
    return mfcc_matrix


# 准备数据
def prepare_data(tom_files, other_files):
    features, labels = [], []
    for file_path in tom_files:
        features.append(extract_features(file_path))
        labels.append(1)
    for file_path in other_files:
        features.append(extract_features(file_path))
        labels.append(0)
    return features, labels


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, num_classes)
        )

    def forward(self, x):
        predict = self.network(x)
        return predict


# 训练神经网络替换原来的train_svm函数
def train_nn(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
    X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
    y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')

    # 创建模型
    input_size = X_train.shape[1]
    num_classes = 2
    model = SimpleNN(input_size, num_classes).to('cuda')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 测试模型
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
        print(f"Accuracy: {accuracy * 100:.2f}%")

    return model, scaler


# 判断是不是Tom
def is_tom(nn_model, scaler, file_path):
    features = extract_features(file_path)
    features_std = scaler.transform([features])
    features_std = torch.tensor(features_std, dtype=torch.float32).to('cuda')
    outputs = nn_model(features_std)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item() == 1


if __name__ == '__main__':
    torch.set_default_device('cuda')

    tom_files = [os.path.join(tom_folder, f)
                 for f in os.listdir(tom_folder)]
    other_files = [os.path.join(not_tom_folder, f)
                   for f in os.listdir(not_tom_folder)]

    # 准备数据并训练神经网络
    features, labels = prepare_data(tom_files, other_files)
    nn_model, scaler = train_nn(features, labels)

    # 使用训练好的模型检测新的音频文件
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]
    for wav_path in test_files:
        if is_tom(nn_model, scaler, wav_path):
            print("Yes " + os.path.basename(wav_path))
            shutil.copy2(wav_path, '识别结果')
        else:
            print("No " + os.path.basename(wav_path))
