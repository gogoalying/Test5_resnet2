import os
import torch
import torch.nn as nn
from model import resnet34
from torch.utils import data

def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\pytorch_classification\Test5_resnet\resnet34-333f7ec4.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    #
    # # option1
    # net = resnet34()
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    # 选择设备
    device = torch.device("cpu")

    # 加载预训练权重
    model_weight_path = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\pytorch_classification\Test5_resnet\resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # 选项1
    net = resnet34()

    # 将模型移动到设备上
    net.to(device)

    # 加载权重
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 改变fc层结构
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    # option2
    # net = resnet34(num_classes=5)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
