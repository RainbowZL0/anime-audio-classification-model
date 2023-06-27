from torch.utils.data import DataLoader

from a001_preprocess import *

"""
待完善：
1. 先划分训练集和验证集，然后再做z score norm. 验证集要用训练集的mean和std做。后续的测试集也是。
2. 分别实现ANN和CNN两个子类。
3. ANN加入batch normalization层。
"""


class MyNN(torch.nn.Module):
    target_class_dataset_folder = TARGET_CLASS_DATASET_FOLDER
    other_class_dataset_folder = OTHER_CLASS_DATASET_FOLDER
    save_model_folder = SAVE_MODEL_FOLDER
    test_wav_folder = TEST_WAV_FOLDER
    has_normalization = HAS_NORMALIZATION
    learning_rate = LR
    train_validation_ratio = TRAIN_VALIDATION_RATIO
    batch_size = BATCH_SIZE
    total_epochs = TOTAL_EPOCHS
    save_model_per_epochs = SAVE_MODEL_FOLDER
    validate_per_epochs = VALIDATE_PER_EPOCHS
    n_fft = N_FFT
    hop_length = HOP_LENGTH

    dim_t: int = None
    dim_t_mean: bool = None

    criterion = None
    optimizer = None

    train_dataloader = None
    validation_dataloader = None
    test_dataloader = None

    def __init__(self):
        super().__init__()
        if self.has_normalization:  # 如果选择启用normalization，则需要记录一下样本集的均值和方差向量。
            self.mean = None
            self.std = None

    def initialize_nn(self):
        dataset_list, label_list = preprocess_training_set(dim_t_0=self.dim_t, get_t_mean=self.dim_t_mean)
        dataset_tensor, label_tensor = torch.stack(dataset_list), torch.tensor(label_list)

        if self.has_normalization:
            dataset_tensor, self.mean, self.std = z_score_normalization(dataset_0=dataset_tensor,
                                                                        mean=self.mean,
                                                                        std=self.std)

        dataset = MyDataset(dataset_tensor, label_tensor)

        train_data_len = int(self.train_validation_ratio * len(dataset))
        val_data_len = len(dataset) - train_data_len
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_data_len, val_data_len],
                                                                   generator=torch.Generator(device=MY_DEVICE))
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           generator=torch.Generator(device=MY_DEVICE),  # 指定随机数生成器
                                           drop_last=True)
        self.validation_dataloader = DataLoader(dataset=val_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                generator=torch.Generator(device=MY_DEVICE),
                                                drop_last=True)
        self.train()  # 设定为train模式

    def train_and_save_model(self):
        for current_epoch in range(1, self.total_epochs + 1):
            # 每训练一整个epoch后输出一次loss值
            loss_i = self.train_for_one_epoch()
            print(f"epoch = {current_epoch} loss = {loss_i}")

            # 每隔指定epoch后保存一次模型,保存时计算在验证集的识别率
            if current_epoch % self.save_model_per_epochs == 0:
                # 保存一次模型
                self.save_model(current_epoch)
                # 计算这次的模型在验证集上的识别率
                self.validate(current_epoch)
            # 虽然不是该保存的轮数, 但是指定了要计算识别率的轮数，也要验证
            elif current_epoch % self.validate_per_epochs == 0:
                self.validate(current_epoch)

    def train_for_one_epoch(self):
        loss = 0
        for batch_data, batch_labels in self.train_dataloader:
            # 前向传播
            predict = self(batch_data)
            # 计算loss
            loss = self.criterion(predict, batch_labels)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 如果在这里添一句 torch.cuda.empty_cache() 可能减少显存占用。但是实测没什么效果。
            # 主要是因为backward反向传播那一步才是最容易爆显存的。
        return loss.item()

    def save_model(self, current_epoch):
        # 若输出文件夹不存在，则创建
        if not os.path.exists(self.save_model_folder):
            os.makedirs(self.save_model_folder)
        # 命名模型文件
        save_model_name = f'epoch{current_epoch}.pth'
        save_model_path = os.path.join(self.save_model_folder, save_model_name)
        # 保存
        torch.save(self.state_dict(), save_model_path)
        # 输出已保存提示
        print(f"save model to {save_model_path}")

    def validate(self, current_epoch):
        self.eval()  # 设为评估模式。作用是关掉网络中的一些层，例如dropout
        correct_num = 0
        total_num = 0

        with torch.no_grad():
            for batch_data, batch_labels in self.validation_dataloader:  # batch_data的形状为(batch_size, mfcc张量)
                # 预测这个batch中每个样本的类别
                predict = self(batch_data)
                _, predict_class = torch.max(predict.data, 1)  # predict_class是一个预测位置的张量
                # 计算预测正确的样本数量
                for i in range(batch_labels.size(0)):
                    if predict_class[i] == batch_labels[i]:  # 若预测和实际的相同，则说明预测正确
                        correct_num += 1
                # 计算目前为止的总样本数。累加
                total_num += batch_data.size(0)
        # 都预测完毕后，返回正确率
        accuracy = correct_num / total_num
        print("epoch = {} validation_acc = {:.3%}".format(current_epoch, accuracy))
