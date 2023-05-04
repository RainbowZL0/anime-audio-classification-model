import os.path
from pydub import AudioSegment

input_txt_folder0 = "segment_length_txt"
input_wav_folder0 = "分集降噪后的wav"
output_wav_folder0 = "切分分集降噪后的wav"


def cut(input_txt_folder, input_wav_folder, output_wav_folder):
    # 先判断输出文件夹是否存在
    if not os.path.exists(output_wav_folder):
        os.makedirs(output_wav_folder)
    # 开始
    input_txt_path_list = [os.path.join(input_txt_folder, f) for f in os.listdir(input_txt_folder)]
    for input_txt_path in input_txt_path_list:
        input_txt_name_and_extension = os.path.basename(input_txt_path)
        input_txt_name = os.path.splitext(input_txt_name_and_extension)[0]
        # 找到对应的wav 然后读入
        input_wav_name_and_extension = input_txt_name + '.wav'
        input_wav_path = os.path.join(input_wav_folder, input_wav_name_and_extension)
        audio = AudioSegment.from_file(input_wav_path, format="wav")
        # 给这个将要被切分的wav准备一个文件夹，存放输出的wav
        new_wav_folder_name = input_txt_name
        new_wav_folder_path = os.path.join(output_wav_folder, new_wav_folder_name)
        if not os.path.exists(new_wav_folder_path):
            os.makedirs(new_wav_folder_path)
        # 读入txt，开始切分
        with open(input_txt_path, 'r', encoding='utf-8') as input_txt:
            segment_cnt = 1  # 已经切分了多少个segment。用于构造输出wav的文件名
            sum_end_time = 0  # 总结束时间，每次切分后更新。用于根据length计算各自的start和end
            line = input_txt.readline()
            while line:
                # 计算新的一个segment的start和end
                segment_length_ms = int(line)
                if segment_length_ms <= 0:  # 如果length<=0，则是没有意义的输入。为了程序健壮性考虑，先排除这个情况
                    continue
                start_time = sum_end_time
                end_time = start_time + segment_length_ms
                # 切分音频
                extracted_audio = audio[start_time:end_time]
                # 构造输出音频的文件名 然后输出
                output_wav_name = convert_to_order(segment_cnt) + '.wav'
                output_wav_path = os.path.join(new_wav_folder_path, output_wav_name)
                extracted_audio.export(output_wav_path, format="wav")

                # 收尾工作 为下一行(下一个segment)做准备
                segment_cnt = segment_cnt + 1  # 更新segment计数
                sum_end_time = end_time  # 更新总结束时间
                line = input_txt.readline()  # 读取下一行


def convert_to_order(INT):
    if 1 <= INT <= 9:
        return '00' + str(INT)
    elif 10 <= INT <= 99:
        return '0' + str(INT)
    else:
        return str(INT)


if __name__ == '__main__':
    cut(input_txt_folder0, input_wav_folder0, output_wav_folder0)
