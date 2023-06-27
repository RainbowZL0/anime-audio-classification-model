import math
import os

import torch
import torchaudio
from torch.utils.data import Dataset
from a000_configuration import *


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


def preprocess_training_set(dim_t_0=None, get_t_mean=False):
    """
    读取训练集的wav文件。
    :return: dataset_list每项都是一个tensor，是样本提取的mfcc。label_list每项都是整数1或0
    """
    torch.set_default_device(MY_DEVICE)

    # 获取文件list
    dataset_class_1, label_class_1, sample_rate = load_wav_data(TARGET_CLASS_DATASET_FOLDER, 1)
    dataset_class_0, label_class_0, _ = load_wav_data(OTHER_CLASS_DATASET_FOLDER, 0)

    # wav_list提取特征为mfcc_list
    dataset_class_1 = extract_features_for_wav_list(dataset_class_1, N_FFT, HOP_LENGTH, dim_t_0=dim_t_0, get_t_mean=get_t_mean)
    dataset_class_0 = extract_features_for_wav_list(dataset_class_0, N_FFT, HOP_LENGTH, dim_t_0=dim_t_0, get_t_mean=get_t_mean)

    # combine
    dataset_list = dataset_class_1 + dataset_class_0
    label_list = label_class_1 + label_class_0

    return dataset_list, label_list


def load_wav_data(folder_path_0, label_0=None):
    """
    总共返回三项，wav_path_list label_list sample_rate
    正样本label=1. 反样本label=0，训练集需要指定1或0。 未指定则为None，可以在调用时忽略返回的label_list，例如测试集没有标签。
    :param folder_path_0: 要读取的wav文件所在的文件夹
    :param label_0: 这些wav文件的标签。如果是训练集，则选1或0。测试集选None，或者省略指定该参数。默认为None。
    :return: wav_path_list, label_list, sample_rate
    """
    wav_path_list = [os.path.join(folder_path_0, wav_name) for wav_name in os.listdir(folder_path_0)]
    if label_0 is not None:
        label_list = [label_0] * len(wav_path_list)  # 一个label值重复填充
    else:  # else是测试集，没标签。此时生成元素值等于index的list
        label_list = [k for k in range(len(wav_path_list))]
    # 获取sample_rate
    _, sample_rate_0 = torchaudio.load(wav_path_list[0])
    return wav_path_list, label_list, sample_rate_0


def extract_features_for_wav_list(wav_path_list, n_fft_0, hop_length_0, dim_t_0=None, get_t_mean=False):
    """
    dim_t指时间轴延长至多少长度。若未指定，则默认为0，代表不延长。
    若get_mean为True，则对三维矩阵的mfcc取平均为n_mfcc长度的向量。
    :param wav_path_list: 一个列表，每项是一个wav_path
    :param n_fft_0: 一次fft所用的采样点数。建议使用全局变量2048。
    :param hop_length_0: fft滑动窗口的采样点数。建议使用全局变量1024。
    :param dim_t_0: 是否需要填充时域为相同的长度。
    :param get_t_mean: 是否对时域维度取平均。
    :return: mfcc_list，一个列表，每项是一个mfcc特征矩阵。
    """
    mfcc_list = []
    for wav in wav_path_list:
        mfcc_matrix = extract_features_for_one_file(wav, n_fft_0, hop_length_0, dim_t_0, get_t_mean)
        mfcc_list.append(mfcc_matrix)
    return mfcc_list


def extract_features_for_one_file(wav_path, n_fft_0, hop_length_0, dim_t_0=None, get_mean=False):
    """
    为了解耦合而重写了。只提取一个文件的特征并返回mfcc矩阵，形状是(channels, n_mfcc, dim_t)。
    dim_t指时间轴延长至多少长度。若未指定，则默认为0，代表不延长。
    若get_mean为True，则对三维矩阵的mfcc取平均为n_mfcc长度的向量。
    :param wav_path:
    :param n_fft_0:
    :param hop_length_0:
    :param dim_t_0:
    :param get_mean:
    :return: 从一个wav文件提取的mfcc矩阵
    """
    # 加载文件
    wav_file, sample_rate_0 = torchaudio.load(wav_path)
    wav_file = wav_file.to(MY_DEVICE)
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


# 让mfcc矩阵的t维度的分量数量统一。也就是将所有样本的时长都延长到相同。暂时只有CNN会用到。
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


def z_score_normalization(dataset_0, mean=None, std=None):
    """
    在输入网络之前做数据集归一化。z-score是一种归一化方法。默认使用这个。
    如果没有指定mean和std，就是训练集。如果指定了mean和std，说明是验证集，应该传入训练集上已经算出的mean和std值作为参数。
    :param dataset_0: 被归一化的数据集。类型是纯tensor。如果要传入list，应该先torch.stack()转为tensor。
    :param mean: 均值
    :param std: 方差
    :return: 归一化后的tensor，均值，方差
    """
    if mean is None and std is None:
        mean = torch.mean(dataset_0, dim=0, keepdim=True)
        std = torch.std(dataset_0, dim=0, keepdim=True)
    dataset_0 = (dataset_0 - mean) / std
    dataset_norm = torch.nan_to_num(dataset_0, nan=0.0)  # 如果有not a number，都改为0
    return dataset_norm, mean, std


def max_min_normalization(dataset_0):  # max-min是另一种归一化方法。默认不用。
    min = torch.min(dataset_0, dim=0, keepdim=True)
    max = torch.max(dataset_0, dim=0, keepdim=True)
    dataset_0 = 2 * (dataset_0 - min) / (max - min) - 1
    return dataset_0
