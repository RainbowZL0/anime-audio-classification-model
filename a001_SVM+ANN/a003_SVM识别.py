import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import shutil
import librosa.feature

# 训练集目录，分为tom的，不是tom的。tom就是指目标角色。文件夹内应该有100个以上的wav文件，每个3s左右
tom_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\训练集\\钉宫语音'
not_tom_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\训练集\\其他语音'

# 待判断是不是tom语音的目录
test_folder = 'D:\\_Search\\Desktop2\\钉宫理惠素材\\SVM\\测试集'


# 提取MFCC特征 mel频率
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_best')

    # n_mfcc是特征数量。40已经算比较大了，更大可能过拟合。
    # n_fft 短时傅里叶变换的段落长度，单位是采样点数。如果更注重短时间内的动态特征，应该调大一点。如果更注重频率特征，应该降至1024
    # hop_length 短时傅里叶变换的窗口滑动长度，单位是采样点数。一般设为n_fft的一半或更小
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=2048, hop_length=1024)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


# 准备数据
def prepare_data(tom_files, other_files):
    features, labels = [], []
    for file_path in tom_files:
        features.append(extract_features(file_path))
        labels.append(1)
    for file_path in other_files:
        features.append(extract_features(file_path))
        labels.append(0)
    return np.array(features), np.array(labels)


# 训练SVM模型
def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    svm_model = SVC(kernel='rbf', C=20, gamma='scale')  # C值越高，分类越严格，默认为10
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return svm_model, scaler


# 判断是不是Tom
def is_tom(svm_model, scaler, file_path):
    features = extract_features(file_path)
    features_std = scaler.transform([features])
    prediction = svm_model.predict(features_std)
    return prediction[0] == 1


if __name__ == '__main__':
    tom_files = [os.path.join(tom_folder, f)
                 for f in os.listdir(tom_folder)]
    other_files = [os.path.join(not_tom_folder, f)
                   for f in os.listdir(not_tom_folder)]

    # 准备数据并训练SVM模型
    features, labels = prepare_data(tom_files, other_files)
    svm_model, scaler = train_svm(features, labels)

    # 使用训练好的模型检测新的音频文件
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder)]
    for wav_path in test_files:
        if is_tom(svm_model, scaler, wav_path):
            print("Yes " + os.path.basename(wav_path))
            shutil.copy2(wav_path, '识别结果')
        else:
            print("No " + os.path.basename(wav_path))
