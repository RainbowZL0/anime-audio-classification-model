import math
import os
import torch
import torchaudio
from __init__ import *
from torch.utils.data import Dataset
from a001_main训练模型 import *


class MyDataset(Dataset):
    def __init__(self, dataset_0, labels_0):
        self.data = dataset_0
        self.labels = labels_0

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)


# class MyDatasetCombineMode(Dataset):
#     def __init__(self, dataset_A, dataset_B, labels_A, labels_B):
#         dataset_tensor = torch.cat((dataset_A, dataset_B), dim=0)  # 数据集处理：两类的样本合并在一起。
#         # dataset_normalization = z_score_normalization(dataset_tensor)  # 归一化z-score
#         self.data = dataset_tensor
#         self.labels = torch.cat((labels_A, labels_B), dim=0)  # 两类的标签也合并。
#
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.labels[index]
#         return x, y
#
#     def __len__(self):
#         return len(self.labels)


# 总共返回三项，wav_path_list label_list sample_rate
# 正样本label=1. 反样本label=0. 未指定则为None. 调用本函数时可以选择忽略返回的label_list，例如测试集没有标签。
def load_wav_data(folder_path_0, label_0=None):
    wav_path_list = [os.path.join(folder_path_0, wav_name) for wav_name in os.listdir(folder_path_0)]
    if label_0 is not None:
        label_list = [label_0] * len(wav_path_list)
    else:  # 说明是测试集，没标签。生成元素值等于下标的list
        label_list = [k for k in range(len(wav_path_list))]
    # 获取sample_rate
    _, sample_rate_0 = torchaudio.load(wav_path_list[0])
    return wav_path_list, label_list, sample_rate_0


# dim_t指时间轴延长至多少长度。若未指定，则默认为0，代表不延长。
# 若get_mean为True，则对三维矩阵的mfcc取平均为n_mfcc长度的向量。
def extract_features_for_wav_list(wav_path_list, n_fft_0, hop_length_0, dim_t_0=None, get_mean=False):
    mfcc_list = []
    for wav in wav_path_list:
        mfcc_matrix = extract_features_for_one_file(wav, n_fft_0, hop_length_0, dim_t_0, get_mean)
        mfcc_list.append(mfcc_matrix)
    return mfcc_list


# 为了解耦合而重写。只提取一个文件的特征并返回mfcc矩阵，形状是(channels, n_mfcc, dim_t)。
# dim_t指时间轴延长至多少长度。若未指定，则默认为0，代表不延长。
# 若get_mean为True，则对三维矩阵的mfcc取平均为n_mfcc长度的向量。
def extract_features_for_one_file(wav_path, n_fft_0, hop_length_0, dim_t_0=None, get_mean=False):
    # 加载文件
    wav_file, sample_rate_0 = torchaudio.load(wav_path)
    wav_file = wav_file.to(my_device)
    # 定义mfcc提取特征的参数，然后转换
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate_0,
        n_mfcc=40,
        melkwargs={"n_fft": n_fft_0, "hop_length": hop_length_0}
    )
    wav_mfcc = mfcc_transform(wav_file)
    # 是否需要延长dim_t
    if dim_t_0 is not None:
        wav_mfcc = unify_dim_t(wav_mfcc, dim_t_0)
    # 是否要取平均为向量
    if get_mean is True:
        wav_mfcc = torch.mean(wav_mfcc, dim=0)
        wav_mfcc = torch.mean(wav_mfcc, dim=1)
    return wav_mfcc


# 让mfcc矩阵的t维度的分量数量统一
def unify_dim_t(mfcc_matrix_0, dim_t):
    zeros_length = dim_t - mfcc_matrix_0.shape[2]  # 要把第3维延长多少长度的0
    zeros_matrix = torch.zeros(mfcc_matrix_0.shape[0], mfcc_matrix_0.shape[1], zeros_length)  # 前两维的长度一样。最后一维延长
    unified_matrix = torch.cat((mfcc_matrix_0, zeros_matrix), dim=2)  # 执行延长操作
    return unified_matrix  # 返回延长后的mfcc特征矩阵


# 计算目标dim_t的大小
def calcu_dim_t(fft_tuple, sample_rate_0):
    # 先转换为浮点
    float_tuple = tuple(float(x) for x in fft_tuple)
    n_fft_0, hop_length_0, duration_0 = float_tuple
    intended_num_samples = sample_rate_0 * duration_0  # 总采样点数
    dim_t = (intended_num_samples - n_fft_0) / hop_length_0  # 计算时间轴的帧数(fft次数)
    dim_t = math.ceil(dim_t)
    return dim_t


# 在输入网络之前做数据集归一化
def z_score_normalization(dataset_0):
    mean = torch.mean(dataset_0, dim=0, keepdim=True)
    std = torch.std(dataset_0, dim=0, keepdim=True)
    dataset_0 = (dataset_0 - mean) / std
    dataset_norm = torch.nan_to_num(dataset_0, nan=0.0)
    return dataset_norm

def max_min_normalization(dataset_0):
    min = torch.min(dataset_0, dim=0, keepdim=True)
    max = torch.max(dataset_0, dim=0, keepdim=True)
    dataset_0 = 2 * (dataset_0-min) / (max-min) - 1
    return dataset_0
