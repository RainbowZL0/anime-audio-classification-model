MY_DEVICE = 'cuda'

TARGET_CLASS_DATASET_FOLDER = "./训练集/钉宫语音"  # 目标类wav数据所在文件夹
OTHER_CLASS_DATASET_FOLDER = "./训练集/其他语音"  # 其他类wav所在文件夹
SAVE_MODEL_FOLDER = "./saved_model"  # 网络模型保存到文件夹
TEST_WAV_FOLDER = "./测试集"  # 待分类的wav所在文件夹
RESULT_FOLDER = "./分类结果"

HAS_NORMALIZATION = True
LR = 0.01
TRAIN_VALIDATION_RATIO = 0.8
BATCH_SIZE = 10

TOTAL_EPOCHS = 100000
SAVE_MODEL_PER_EPOCHS = 200
VALIDATE_PER_EPOCHS = 10

N_FFT = 2048
HOP_LENGTH = 1024




