import os.path

import torch
from torch import optim
from __init__ import *

from a001_网络结构_v0_旧ANN模型 import *
from a001_网络结构_v1_2亿参数 import *
from a001_数据集生成 import *
from a001_网络结构_v2_2亿参数_卷积通道减半_核增大 import *
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

# 目标类wav数据所在文件夹
target_class_dataset_folder = "./训练集/钉宫语音"
# 其他类wav所在文件夹
other_class_dataset_folder = "./训练集/其他语音"
# 模型保存到文件夹
save_model_folder = "./saved_model"
# 待分类的wav所在文件夹
test_wav_folder = ""

# 计算设备选择
my_device = 'cuda'

# sample_rate
sample_rate_global = 44100  # 由于会影响到网络结构，而网络结构暂时是写死的，所以这一组值目前还不能随意调整。
# 用mfcc()提取特征的参数 （也不能随意调整）
n_fft = 2048
hop_length = 1024
length_duration_to = 5  # 将每个wav样本补0到多长时间(秒)

# 训练参数。可以任意调整。
batch_size = 3
total_epoch = 10000
learning_rate = 0.0001
save_model_per_epochs = 999999999  # 每隔多少个epoch，保存一次模型
validate_per_epochs = 5  # 每隔多少个epoch，在验证集上测试一次性能

# 训练集的比例。剩下的作为验证集
train_data_ratio = 0.8


def train_cnn():
    # 指定设备
    torch.set_default_device(my_device)

    # 获取文件list
    dataset_class_1, label_class_1, sample_rate = load_wav_data(target_class_dataset_folder, 1)
    dataset_class_0, label_class_0, _ = load_wav_data(other_class_dataset_folder, 0)

    # 计算mfcc张量的时间轴预计要延长到多少
    dim_t = calcu_dim_t((n_fft, hop_length, length_duration_to), sample_rate)

    # wav_list提取特征为mfcc_list
    dataset_class_1 = extract_features_for_wav_list(dataset_class_1, n_fft, hop_length, dim_t)
    dataset_class_0 = extract_features_for_wav_list(dataset_class_0, n_fft, hop_length, dim_t)

    # 两类的mfcc和label都转换为张量tensor
    dataset_class_1, label_class_1 = torch.stack(dataset_class_1), torch.tensor(label_class_1)
    dataset_class_0, label_class_0 = torch.stack(dataset_class_0), torch.tensor(label_class_0)



    # norm
    total_dataset = torch.cat((dataset_class_1, dataset_class_0), dim=0)
    total_label = torch.cat((label_class_1, label_class_0))
    total_dataset = z_score_normalization(total_dataset)
    # 打包
    total_dataset = MyDataset(total_dataset, total_label)



    # # 两类的数据和标签分别打包成两个Dataset类的对象。
    # dataset_and_labels_class1 = MyDataset(dataset_class_1, label_class_1)
    # dataset_and_labels_class0 = MyDataset(dataset_class_0, label_class_0)
    #
    # # 合并两个类的dataset_and_labels
    # total_dataset = ConcatDataset([dataset_and_labels_class0, dataset_and_labels_class1])



    # 按照指定的比例 划分出训练集和验证集
    train_size = int(train_data_ratio * len(total_dataset))
    validation_size = len(total_dataset) - train_size
    train_dataset, validation_dataset = random_split(total_dataset, [train_size, validation_size],
                                                     generator=torch.Generator(
                                                         device=my_device))  # 随机数生成器generator要放在my_device上，否则报错设备不同
    # 按照batch_size随机划分batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device=my_device))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator(device=my_device))

    # 开始训练
    model = CNNv2()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_and_save_model(model_0=model,
                         criterion_0=criterion,
                         optimizer_0=optimizer,
                         train_dataloader_0=train_dataloader,
                         validation_dataloader_0=validation_dataloader,
                         total_epoch_0=total_epoch,
                         save_model_per_epochs_0=save_model_per_epochs,
                         validate_per_epochs_0=validate_per_epochs)


def train_and_save_model(model_0, criterion_0, optimizer_0, train_dataloader_0, validation_dataloader_0, total_epoch_0,
                         save_model_per_epochs_0, validate_per_epochs_0):
    for current_epoch in range(1, total_epoch_0 + 1):
        # 训练一整个epoch后输出一次loss值
        loss_i = train_per_epoch(model_0=model_0,
                                 criterion_0=criterion_0,
                                 optimizer_0=optimizer_0,
                                 dataloader_0=train_dataloader_0)
        print(f"epoch = {current_epoch} loss = {loss_i}")

        # 每隔指定epoch后保存一次模型,保存时计算在验证集的识别率
        if current_epoch % save_model_per_epochs_0 == 0:
            # 保存一次模型
            save_model(current_epoch, model_0)
            # 计算这次的模型在验证集上的识别率
            validate(current_epoch, model_0, validation_dataloader_0)
        # 虽然不是该保存的轮数, 但是指定了要计算识别率的轮数
        elif current_epoch % validate_per_epochs_0 == 0:
            validate(current_epoch, model_0, validation_dataloader_0)


def train_per_epoch(model_0, criterion_0, optimizer_0, dataloader_0):
    loss = 0
    for batch_data, batch_labels in dataloader_0:
        # 前向传播
        predict = model_0(batch_data)
        # 计算loss
        loss = criterion_0(predict, batch_labels)
        # 反向传播
        optimizer_0.zero_grad()
        loss.backward()
        optimizer_0.step()
        # 如果在这里添一句 torch.cuda.empty_cache() 可能减少显存占用
    return loss.item()


def save_model(current_epoch, model_0):
    # 若输出文件夹不存在，则创建
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)
    # 命名模型文件
    save_model_name = f'epoch{current_epoch}.pth'
    save_model_path = os.path.join(save_model_folder, save_model_name)
    # 保存
    torch.save(model_0.state_dict(), save_model_path)
    # 输出已保存提示
    print(f"save model to {save_model_path}")


def validate(current_epoch_0, model_0, test_dataloader):
    model_0.eval()  # 设为评估模式。作用是关掉网络中的一些层，例如dropout
    correct_num = 0
    total_num = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_dataloader:  # batch_data的形状为(batch_size, mfcc张量)
            # 预测这个batch中每个样本的类别
            predict = model_0(batch_data)
            _, predict_class = torch.max(predict.data, 1)  # predict_class是一个预测位置的张量
            # 计算预测正确的样本数量
            for i in range(batch_labels.size(0)):
                if predict_class[i] == batch_labels[i]:  # 若预测和实际的相同，则说明预测正确
                    correct_num += 1
            # 计算目前为止的总样本数。累加
            total_num += batch_data.size(0)
    # 都预测完毕后，返回正确率
    accuracy = correct_num / total_num
    print("epoch = {} validation_acc = {:.3%}".format(current_epoch_0, accuracy))


"""-----------------------------------------------------------------------------------------------------
下面是给v0_旧ANN模型用的函数。
"""


def train_ann():
    # 指定设备
    torch.set_default_device(my_device)

    # 获取文件list
    dataset_class_1, label_class_1, sample_rate = load_wav_data(target_class_dataset_folder, 1)
    dataset_class_0, label_class_0, _ = load_wav_data(other_class_dataset_folder, 0)
    # 文件list提取特征为mfcc_list
    dataset_class_1 = extract_features_for_wav_list(dataset_class_1, n_fft, hop_length, get_mean=True)
    dataset_class_0 = extract_features_for_wav_list(dataset_class_0, n_fft, hop_length, get_mean=True)
    # 转换为张量
    dataset_class_1, label_class_1 = torch.stack(dataset_class_1), torch.tensor(label_class_1)
    dataset_class_0, label_class_0 = torch.stack(dataset_class_0), torch.tensor(label_class_0)


    # norm
    total_dataset = torch.cat((dataset_class_1, dataset_class_0), dim=0)
    total_label = torch.cat((label_class_1, label_class_0))
    total_dataset = z_score_normalization(total_dataset)
    # 打包
    total_dataset = MyDataset(total_dataset, total_label)



    # # 两类的数据和标签分别打包成两个Dataset类的对象。
    # dataset_and_labels_class1 = MyDataset(dataset_class_1, label_class_1)
    # dataset_and_labels_class0 = MyDataset(dataset_class_0, label_class_0)
    #
    # # 合并两个类的dataset_and_labels
    # total_dataset = ConcatDataset([dataset_and_labels_class0, dataset_and_labels_class1])




    # 按照指定的比例 划分出训练集和验证集
    train_size = int(train_data_ratio * len(total_dataset))
    validation_size = len(total_dataset) - train_size
    train_dataset, validation_dataset = random_split(total_dataset, [train_size, validation_size],
                                                     generator=torch.Generator(
                                                         device=my_device))  # 随机数生成器generator要放在my_device上，否则报错设备不同
    # 按照batch_size随机划分batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator(device=my_device))
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator(device=my_device))

    # 开始训练
    model = SimpleNN()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_and_save_model_ann(model, criterion, optimizer, train_dataloader, validation_dataloader,
                             total_epoch, save_model_per_epochs, validate_per_epochs)


def train_and_save_model_ann(model_0, criterion_0, optimizer_0, train_dataloader_0, validation_dataloader_0, total_epoch_0,
                             save_model_per_epochs_0, validate_per_epochs_0):
    for current_epoch in range(1, total_epoch_0 + 1):
        # 训练一整个epoch后输出一次loss值
        loss_i = train_per_epoch(model_0=model_0,
                                 criterion_0=criterion_0,
                                 optimizer_0=optimizer_0,
                                 dataloader_0=train_dataloader_0)
        print(f"epoch = {current_epoch} loss = {loss_i}")

        # 每隔指定epoch后保存一次模型,保存时计算在验证集的识别率
        if current_epoch % save_model_per_epochs_0 == 0:
            # 保存一次模型
            save_model(current_epoch, model_0)
            # 计算这次的模型在验证集上的识别率
            validate_ann(current_epoch, model_0, validation_dataloader_0)
        # 虽然不是该保存的轮数, 但是指定了要计算识别率的轮数
        elif current_epoch % validate_per_epochs_0 == 0:
            validate_ann(current_epoch, model_0, validation_dataloader_0)


def validate_ann(current_epoch_0, model_0, test_dataloader):
    model_0.eval()  # 设为评估模式。作用是关掉网络中的一些层，例如dropout
    correct_num = 0
    total_num = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_dataloader:  # batch_data的形状为(batch_size, mfcc张量)
            # 预测这个batch中每个样本的类别
            predict = model_0(batch_data)
            _, predict_class = torch.max(predict.data, 1)  # predict_class是一个预测位置的张量
            # 计算预测正确的样本数量
            for i in range(batch_labels.size(0)):
                if predict_class[i] == batch_labels[i]:  # 若预测和实际的相同，则说明预测正确
                    correct_num += 1
            # 计算目前为止的总样本数。累加
            total_num += batch_data.size(0)
    # 都预测完毕后，返回正确率
    accuracy = correct_num / total_num
    print("epoch = {} validation_acc = {:.3%}".format(current_epoch_0, accuracy))


if __name__ == '__main__':
    train_cnn()
