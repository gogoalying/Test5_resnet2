import os
import torch
import torch.nn as nn
from model import resnet34
from torch.utils import data

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\pytorch_classification\Test5_resnet\resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # # option1
    # net = resnet34()
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    # print(net)
    # 选择设备
    # device = torch.device("cpu")
    # print(device)
    # net = resnet34()
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "resnet34-333f7ec4.pth"
    #
    # if device.type == 'cuda':
    #     net.load_state_dict(torch.load(model_weight_path))
    # else:
    #     net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False
    # # 引入迁移学习61-69
    # # pre_weights = torch.load(model_weight_path, map_location=device)
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    # net.to(device)
    # print(net)
    # for k in pre_weights.keys():
    #     if 'fc' in k:
    #         print(k)

    # option2

    net = resnet34(num_classes=100)
    new_weights = net.state_dict()
    pre_weights = torch.load(model_weight_path)
    print(type(pre_weights))

    for k in pre_weights.keys():
        if k in new_weights.keys() and not k.startswith("fc") and "fc" not in k:
            new_weights[k] = pre_weights[k]
    net.load_state_dict(new_weights)
    print(net)
    params = []
    train_layer = ['fc']
    for name, param in net.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layer):
            print(name)
            params.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    # for k in new_weights.keys():
    #     if 'fc' in k:
    #         print(k)


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
