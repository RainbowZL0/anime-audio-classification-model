import torch

torch.set_default_device('cuda')  # 默认设备
torch.set_default_tensor_type(torch.FloatTensor)  # 默认张量类型


def normalization():
    data1 = torch.tensor([0, 0, 1])
    print(data1.dtype)  # 查看张量类型
    data2 = torch.tensor([1, 1, 2], dtype=torch.float)

    list0 = [data1, data2]
    sums = torch.stack(list0, dim=0)  # stack是创建一个新的维度。而cat是延长某个维度的分量的数量

    # 归一化
    mean = torch.mean(sums, dim=0)
    std = torch.std(sums, dim=0)
    data = (sums - mean) / std

    print(data)


def tensor_padding():
    # 创建一个 2x3 的张量
    x = torch.rand(2, 3, 4)
    print(x)
    print()

    y = torch.zeros(2, 3, 1)

    cat_x = torch.cat((x, y), dim=2)


# 添加一个维度，并且复制填充这个新维度
def un_squeeze_repeat_test():
    a = torch.ones(2, 2)
    b = torch.unsqueeze(a, dim=0)  # dim=0是指在原来的dim=0之前添加一个维度。如果要添加到最后一个维度之后，填入dim=-1
    print(b.shape)

    c = b.repeat(3, 1, 1)  # dim=0上复制3遍，dim=1上复制1遍（不做操作），dim=2上复制一遍
    print(c.shape)


def print_test():
    print("epoch = {} validation_acc = {:.3%}".format("哈哈", 0.8))


if __name__ == '__main__':
    a = torch.tensor([1., 1.])
    b = torch.tensor([3., 2.])
    c = torch.stack((a, b), dim=0)
    mean = torch.mean(c, dim=0, keepdim=True)
    std = torch.std(c, dim=0, keepdim=True)


    norm = (c - mean) / std
    print(norm)
