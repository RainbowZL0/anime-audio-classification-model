import os.path
import shutil

import torch

from a001_网络结构_v1_2亿参数 import *
from a001_main训练模型 import *
from a001_数据集生成 import *

test_file_folder = "./测试集"
loaded_model_path = "./saved_model/epoch200新训练集_norm.pth"

output_folder_class1 = "./分类结果/Class_1"
output_folder_class0 = "./分类结果/Class_0"

my_device_1 = 'cuda'  # 在 a001_main训练模型.py 中也有一个my_device，这两个不冲突，训练和推理时可以选不同的设备。

test_batch_size = 100

# class TestDataset(Dataset):
#     def __init__(self, dataset):
#         self.data = dataset
#
#     def __getitem__(self, index):
#         data = self.data[index]
#         return data
#
#     def __len__(self):
#         return len(self.data)


# 加载待分类的数据集。和训练用的那个load函数只有微小差别。改成了针对单个文件提取，而不是目录下的所有文件。
def load_wav_data_1(wav_path, n_fft_0, hop_length_0, length_duration_to_0):
    # 当所有样本都统一延长至duration时，需要计算的dim_t
    dim_t = calcu_dim_t((n_fft_0, hop_length_0, length_duration_to_0), sample_rate_global)
    mfcc_matrix = extract_features_1(wav_path, n_fft_0=n_fft_0, hop_length_0=hop_length_0, dim_t=dim_t)  # 提取mfcc
    return mfcc_matrix


# 也是和训练用的这个函数基本一样。
def extract_features_1(wav_path, n_fft_0, hop_length_0, dim_t):
    # 读取文件
    wav_file, sample_rate_0 = torchaudio.load(wav_path)
    wav_file = wav_file.to(my_device_1)
    # mfcc计算
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate_0,
        n_mfcc=40,
        melkwargs={'n_fft': n_fft_0, 'hop_length': hop_length_0}
    )
    mfcc_matrix = mfcc_transform(wav_file)  # 返回张量(2, n_mfcc, t)
    # t维度拉长为统一的值
    mfcc_matrix = unify_dim_t(mfcc_matrix, dim_t=dim_t)
    return mfcc_matrix


# 按batch_size推理
def predict_class(model_1, test_dataloader_0, wav_path_list_0):
    model_1.eval()
    with torch.no_grad():
        for batch_data, index_tensor_0 in test_dataloader_0:  # index_tensor是一维向量，存储了每个样本刚读入wav_list时的位置，可以回去找到是哪个样本。不然batch都被打散了。
            predict = model_1(batch_data)
            _, predicted_class = torch.max(predict.data, dim=1)  # predicted_class是一维向量，取值1或0代表了本batch内的分类预测
            for i in range(len(predicted_class)):
                if predicted_class[i] == 1:  # 查看预测向量中的每个类别。分别输出。
                    output_class(wav_path_list_0, index_tensor_0[i], True)
                else:
                    output_class(wav_path_list_0, index_tensor_0[i], False)


def output_class(wav_path_list_0, index, is_class_1):
    wav_path = wav_path_list_0[index]
    wav_name_0 = os.path.basename(wav_path)
    if is_class_1:
        print(f"YES {wav_name_0}")
        save_wav_path = os.path.join(output_folder_class1, wav_name_0)  # 根据类别不同，分别生成不同的保存路径
    else:
        print(f"NO {wav_name_0}")
        save_wav_path = os.path.join(output_folder_class0, wav_name_0)
    shutil.copy2(wav_path, save_wav_path)  # 保存


# 推理
def is_class1(model_1, one_test_data):
    with torch.no_grad():
        predict = model_1(one_test_data)
        _, predicted_class = torch.max(predict.data, dim=0)  # 如果是批次数据，则此处改为1
        print(predict.data)
        if predicted_class.item() == 1:  # 说明是第1类，也就是目标类。如果等于0则为其他人。
            return True
        else:
            return False


def is_class1_ann(model_1, one_test_data):
    with torch.no_grad():
        predict = model_1(one_test_data)
        _, predicted_class = torch.max(predict.data, dim=0)
        # print(predict.data)
        if predicted_class.item() == 1:  # 说明是第1类，也就是目标类。如果等于0则为其他人。
            return True
        else:
            return False


if __name__ == '__main__':
    # 指定设备
    torch.set_default_device(my_device_1)
    # 加载模型，并且设置为评估模式。
    loaded_model = CNNv2()
    # loaded_model = SimpleNN()
    loaded_model.load_state_dict(torch.load(loaded_model_path))
    loaded_model.eval()
    # 检查输出目录是否存在。不存在则创建
    if not os.path.exists(output_folder_class1):
        os.makedirs(output_folder_class1)
    if not os.path.exists(output_folder_class0):
        os.makedirs(output_folder_class0)

    # 加载文件
    wav_path_list, index_list, sample_rate = load_wav_data(test_file_folder, label_0=None)
    # 提取为mfcc_list 然后转换为tensor
    mfcc_data = extract_features_for_wav_list(wav_path_list, n_fft, hop_length, dim_t_0=214, get_mean=False)
    mfcc_tensor = torch.stack(mfcc_data)
    index_tensor = torch.tensor(index_list)

    # 归一化
    mfcc_tensor = z_score_normalization(mfcc_tensor)

    # 封装为数据集。主要是为了利用torch提供的按照batch推理的功能。
    test_dataset = MyDataset(mfcc_tensor, index_tensor)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False)

    predict_class(loaded_model, test_dataloader, wav_path_list)








