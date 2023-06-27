import random
import shutil

from sklearn import svm
from a001_preprocess import *
from a000_configuration import *


class MySVMModel:
    def __init__(self, my_model, has_norm):
        """
        初始化。
        :param my_model: 一个SVM类的实例。
        :param has_norm: boolean，是否启用归一化
        """
        self.model = my_model

        self.normalize = has_norm
        if self.normalize:  # 当训练前归一化启用时，需要记录训练集样本的方差和均值
            self.mean = 0
            self.std = 0

        self.training_dataset_list = None
        self.training_label_list = None
        self.validation_dataset_list = None
        self.validation_label_list = None

    def train_svm(self):
        """
        训练一个svm模型。参数都在a000_configuration.py的全局全量中定义。
        """
        dataset_list, label_list = preprocess_training_set(dim_t_0=None, get_t_mean=True)  # 此时dataset list里元素为tensor
        training_dataset_list, \
            self.training_label_list, \
            validation_dataset_list, \
            self.validation_label_list = split_dataset(dataset_list, label_list)  # 划分训练集和验证集

        if self.normalize:  # 如果选择了归一化
            training_dataset_tensor = torch.stack(training_dataset_list)
            training_dataset_tensor, self.mean, self.std = z_score_normalization(training_dataset_tensor)
            self.training_dataset_list = training_dataset_tensor.tolist()

            # 验证集的归一化，用的mean和std都是训练集上已经算好的。
            validation_dataset_tensor = torch.stack(validation_dataset_list)
            validation_dataset_tensor, _, _ = z_score_normalization(validation_dataset_tensor, self.mean, self.std)
            self.validation_dataset_list = validation_dataset_tensor.tolist()
        else:  # 如果没有启用归一化，就直接转换为cpu上的list。
            # device to cpu. 因为sklearn只支持cpu训练。但是之前的提取特征可以用cuda加速。
            self.training_dataset_list = [tensor.cpu().numpy() for tensor in training_dataset_list]
            self.validation_dataset_list = [tensor.cpu().numpy() for tensor in validation_dataset_list]

        # 一句话训练
        self.model.fit(self.training_dataset_list, self.training_label_list)

    def validate_svm(self):
        """
        训练完成后，在验证集上验证模型识别率
        :return: None
        """
        true_positive = 0  # 预测为正例(positive)，且该预测正确(true)的样本数。这四项构成了概念”混淆矩阵“的定义，用于评估模型效果。
        true_negative = 0  # 预测为反例(negative)，且该预测正确(true)的样本数
        false_positive = 0  # 后两个同理。
        false_negative = 0

        prediction_ndarray = self.model.predict(self.validation_dataset_list)
        for index in range(len(prediction_ndarray)):
            real_value = self.validation_label_list[index]
            predicted_value = prediction_ndarray[index]
            if predicted_value == 1 and real_value == 1:
                true_positive += 1
            elif predicted_value == 1 and real_value == 0:
                false_positive += 1
            elif predicted_value == 0 and real_value == 0:
                true_negative += 1
            elif predicted_value == 0 and real_value == 1:
                false_negative += 1

        total_true = true_positive + true_negative
        total = len(prediction_ndarray)
        accuracy = total_true / total  # 预测对的所有样本数，除以全部样本数，得到总识别率

        print("Validation result:")
        print(f"TP = {true_positive}, TN = {true_negative}, FP = {false_positive}, FN = {false_negative}")
        print("Accuracy = {:.3%}".format(accuracy))
        print()

    def predict_svm(self, test_wav_folder):
        """
        预测一些wav文件的类别。输出信息时，调用静态方法print_prediction()
        :param self: 已经训练好的model
        :param test_wav_folder: 待分类的wav文件所在的文件夹
        """
        wav_path_list, _, _ = load_wav_data(test_wav_folder, label_0=None)
        mfcc_list = extract_features_for_wav_list(wav_path_list, N_FFT, HOP_LENGTH, dim_t_0=None, get_t_mean=True)

        if self.normalize:
            mfcc_tensor = torch.stack(mfcc_list)
            mfcc_tensor, _, _ = z_score_normalization(mfcc_tensor, mean=self.mean, std=self.std)
            mfcc_list = mfcc_tensor.tolist()
        else:
            mfcc_list = [tensor.cpu().numpy() for tensor in mfcc_list]

        prediction_ndarray = self.model.predict(mfcc_list)  # 预测
        MySVMModel.output_prediction(prediction_ndarray, wav_path_list)  # 输出信息

    @staticmethod
    def output_prediction(prediction_ndarray, wav_path_list):
        """
        输出每个wav的类别预测。并且复制两类的结果到指定的输出文件夹。
        :param prediction_ndarray:
        :param wav_path_list:
        """
        #  准备好输出文件夹，如果文件夹已经存在，将会被清空。
        if os.path.exists(RESULT_FOLDER):
            shutil.rmtree(RESULT_FOLDER)
        os.makedirs(RESULT_FOLDER)
        class_0_folder = os.path.join(RESULT_FOLDER, "class_0")
        class_1_folder = os.path.join(RESULT_FOLDER, "class_1")
        os.makedirs(class_0_folder)
        os.makedirs(class_1_folder)

        # 预测，输出预测信息，并且复制音频文件到两类输出文件夹中
        for index, prediction_label in enumerate(prediction_ndarray):
            wav_path = wav_path_list[index]  # 取出wav文件的路径
            wav_name = os.path.basename(wav_path)  # 路径的最后一部分是文件名
            if prediction_label == 1:
                print_info = f'Yes! {wav_name}'
                copy_to_path = os.path.join(class_1_folder, wav_name)
            else:
                print_info = f'No!  {wav_name}'
                copy_to_path = os.path.join(class_0_folder, wav_name)
            shutil.copy2(wav_path, copy_to_path)
            print(print_info)


def split_dataset(dataset_list_0, label_list_0):
    """
    用于随机划分训练集和验证集。
    :param dataset_list_0: 所有数据集
    :param label_list_0: 所有标签集
    :return: 训练集数据，训练集标签，验证集数据，验证集标签。都是list。
    """
    shuffled_indices_list = generate_random_list_indices(len(label_list_0))

    shuffled_dataset = randomly_shuffle_list(dataset_list_0, shuffled_indices_list)
    shuffled_label_list = randomly_shuffle_list(label_list_0, shuffled_indices_list)

    training_set_size = int(len(label_list_0) * TRAIN_VALIDATION_RATIO)

    training_set = shuffled_dataset[:training_set_size]
    training_set_label = shuffled_label_list[:training_set_size]
    validation_set = shuffled_dataset[training_set_size:]
    validation_set_label = shuffled_label_list[training_set_size:]

    return training_set, training_set_label, validation_set, validation_set_label


def generate_random_list_indices(length_0: int):
    indices_list_0 = random.sample(range(length_0), length_0)
    return indices_list_0


def randomly_shuffle_list(list_0, indices_list_0):
    shuffled_list = [list_0[j] for j in indices_list_0]
    return shuffled_list


def start_svm():
    new_model = svm.SVC(kernel='rbf', C=1)  # 可以调整C值。更大的C代表更加拟合数据，增大训练集上的正确率，代价是可能过拟合。
    my_svm_model = MySVMModel(new_model, has_norm=HAS_NORMALIZATION)
    my_svm_model.train_svm()  # 训练
    my_svm_model.validate_svm()  # 验证
    my_svm_model.predict_svm(TEST_WAV_FOLDER)  # 测试


if __name__ == '__main__':
    start_svm()

