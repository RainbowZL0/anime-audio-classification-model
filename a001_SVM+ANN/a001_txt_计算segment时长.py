import os.path

in_txt_folder = "start_end_txt"
out_txt_folder = "segment_length_txt"


def compute_segment_length(input_txt_folder, output_txt_folder):
    # 先判断输出文件夹是否存在。不存在则创建
    if not os.path.exists(output_txt_folder):
        os.makedirs(output_txt_folder)

    # 开始操作。逻辑是读取一行，计算差值，写入
    input_txt_files_path = [os.path.join(input_txt_folder, f) for f in os.listdir(input_txt_folder)]
    for input_txt_path in input_txt_files_path:
        input_txt_name = os.path.basename(input_txt_path)
        output_txt_name = input_txt_name
        output_txt_path = os.path.join(output_txt_folder, output_txt_name)

        with open(input_txt_path, 'r', encoding='utf-8') as input_txt, open(output_txt_path, 'w',
                                                                            encoding='utf-8') as output_txt:
            line = input_txt.readline()
            line_cnt = 1
            while line:
                words = line.split(',')
                segment_length_ms = convert_to_ms(words[1]) - convert_to_ms(words[0])
                if line_cnt == 1:
                    write_words = str(segment_length_ms)
                else:
                    write_words = f"\n{str(segment_length_ms)}"
                output_txt.write(write_words)

                line_cnt = line_cnt + 1
                line = input_txt.readline()


def convert_to_ms(input_string):
    # 先转换成秒
    second_string = input_string.split(':')
    second = float(second_string[0]) * 3600 + float(int(second_string[1]) * 60) + float(second_string[2])
    # 再转换成毫秒，最后转换到整数。
    ms = int(second * 1000)
    return ms


if __name__ == '__main__':
    compute_segment_length(in_txt_folder, out_txt_folder)
