import os
import sys
import json
import torch
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from model import resnet34
from dataset import TrainM, TestM
from torch.nn import functional as F
from Ascension import Ascension
from test import get_mae_age


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = TrainM(data_transform["train"])


    # # train_num = len(train_dataset)
    train_num = len(train_dataset)
    # # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = TestM(data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    model_weight_path = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\pytorch_classification\Test5_resnet\resnet34-333f7ec4.pth"

    # # for param in net.parameters():
    # #     param.requires_grad = False
    # #引入迁移学习61-69
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    # # net.fc1 = nn.Linear(in_channel, 4)
    # net.to(device)

    net = resnet34(num_classes=5)
    new_weights = net.state_dict()
    pre_weights = torch.load(model_weight_path)
    new_weights = net.state_dict()
    for k in pre_weights.keys():
        if k in new_weights.keys() and not k.startswith("fc") :
            new_weights[k] = pre_weights[k]
    # 加载权重
    net.load_state_dict(new_weights)
    # 将模型移动到指定设备上
    net.to(device)
    # construct an optimizer
    params = []
    train_layer = ['fc']
    for name, param in net.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layer):
            print(name)
            params.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.001)
    # 初始化梯度列表
    gradients_fc = []
    gradients_fc1 = []
    epochs = 3
    best_mae = 0.0
    save_path = 'resnet34-333f7ec4.pth.pth'
    train_steps = len(train_loader)
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.cuda(non_blocking=True)
            target = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            # logits = net(images.to(device))
            mae = 0
            # rank = torch.tensor([i for i in range(101)]).cuda()

            output, output1 = net(images)
            loss = loss_function(output, target)
            loss1 = loss_function(output1, target)
            total_loss = loss + loss1
            total_loss.backward()
            running_loss += loss.item()

            # print("原始张量{}\n扩张后张量{}\nloss:{}".format(output, output1,loss))
            # individual_tensors = list(output.chunk(output.size(0), dim=0))
            # result_list = []
            # for tensor in individual_tensors:
            #     tensor = tensor.view(-1)
            #     b = Ascension(tensor) # 经过函数扩张到想要的类别
            #     result_list.append(b)
            #     # 将结果保存到一个新的二维张量中
            # output1 = torch.stack(result_list)
            # 提取并绘制fc层和fc1层的梯度曲线

            # 检查fc1层的梯度
            for name, param in net.named_parameters():
                if 'fc1' in name:
                    print('fc1层的梯度：', param.grad)
                else:
                    print("无fc1梯度")
            optimizer.step()
        # plt.plot(gradients_fc, label='fc')
        # plt.plot(gradients_fc1, label='fc1')
        # plt.xlabel('Epoch')
        # plt.ylabel('Gradient')
        # plt.title('Gradient Visualization')
        # plt.legend()
        # plt.show()
    #         # pred = torch.sum(output1 * rank, dim=1)
    #         mae = torch.sum(torch.abs(output - target.unsqueeze(1)), dim=1).mean()
    #         # print statistics
    #         running_loss += loss.item()
    #
    #         train_bar.desc = "train epoch[{}/{}] loss:{:.3f} mae:{:.3f}".format(epoch + 1,
    #                                                                  epochs,
    #                                                                  loss,mae)
    #
    #         print(output)
    #         print(target.shape)
    #         print(output.shape)
    #     # validate
    #     net.eval()
    #     with torch.no_grad():
    #         val_bar = tqdm(validate_loader, file=sys.stdout)
    #         for val_data in val_bar:
    #             val_images, val_labels = val_data
    #             output = net(val_images.to(device))
    #             output1 = F.softmax(output, dim=1)
    #
    #             # rank = torch.Tensor([i for i in range(101)]).cuda()
    #             # pred = torch.sum(output * rank, dim=1)
    #             images = images.cuda(non_blocking=True)
    #             target = val_labels.cuda(non_blocking=True)
    #             # loss = loss_function(outputs, test_labels)
    #             mae = 0
    #             mae = torch.sum(torch.abs(output1 - target.unsqueeze(1)), dim=1).mean()
    #
    #             val_bar.desc = "valid epoch[{}/{}, mae:[{}]]".format(epoch + 1,
    #                                                        epochs,mae)
    #
    #             print(output1.shape)
    #             print(val_labels.shape)
    #             print(output.shape)
    #     # if mae < best_mae:
    #     #     best_mae = mae
    #     #     torch.save(net.state_dict(), save_path)
    #
    # print('Finished Training')


if __name__ == '__main__':
    main()


#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#     print(device)
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#         "val": transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#
#     # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
#     # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
#     # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
#
#     train_dataset = TrainM(data_transform["train"])
#
#
#     # # train_num = len(train_dataset)
#     train_num = len(train_dataset)
#     # # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
#     # flower_list = train_dataset.class_to_idx
#     # cla_dict = dict((val, key) for key, val in flower_list.items())
#     # # write dict into json file
#     # json_str = json.dumps(cla_dict, indent=4)
#     # with open('class_indices.json', 'w') as json_file:
#     #     json_file.write(json_str)
#
#     batch_size = 16
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size, shuffle=True,
#                                                num_workers=nw)
#
#     validate_dataset = TestM(data_transform["val"])
#     val_num = len(validate_dataset)
#     validate_loader = torch.utils.data.DataLoader(validate_dataset,
#                                                   batch_size=batch_size, shuffle=False,
#                                                   num_workers=nw)
#
#     print("using {} images for training, {} images for validation.".format(train_num,
#                                                                            val_num))
#
#     net = resnet34()
#     # load pretrain weights
#     # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
#     model_weight_path = "resnet34-333f7ec4.pth"
#
#     if device.type == 'cuda':
#         net.load_state_dict(torch.load(model_weight_path))
#     else:
#         net.load_state_dict(torch.load(model_weight_path, map_location=device))
#     for param in net.parameters():
#         param.requires_grad = False
#     # 引入迁移学习61-69
#
#     # change fc layer structure
#     in_channel = net.fc.in_features
#     net.fc = nn.Linear(in_channel, 5)
#     net.to(device)
#
#
#     # net = resnet34(num_classes=5)
#     # model_weight_path = "resnet34-333f7ec4.pth"
#     # new_weights = net.state_dict()
#     # pre_weights = torch.load(model_weight_path, map_location=device)
#     # new_weights = net.state_dict()
#     # if device.type == 'cuda':
#     #     net.load_state_dict(torch.load(model_weight_path))
#     # else:
#     #     net.load_state_dict(torch.load(model_weight_path, map_location=device))
#     # for k in pre_weights.keys():
#     #    if k in new_weights.keys() and not k.startswith("fc") :
#     #         new_weights[k] = pre_weights[k]
#     # net.load_state_dict(new_weights)
#     # net.to(device)
#     # construct an optimizer
#     params = [p for p in net.parameters() if p.requires_grad]
#     optimizer = optim.Adam(params, lr=0.0001)
#
#     epochs = 1
#     best_mae = 0.0
#     save_path = 'resnet34-333f7ec4.pth.pth'
#     train_steps = len(train_loader)
#     # define loss function
#     loss_function = nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         # train
#         net.train()
#         running_loss = 0.0
#         train_bar = tqdm(train_loader, file=sys.stdout)
#         for step, data in enumerate(train_bar):
#             images, labels = data
#             images = images.cuda(non_blocking=True)
#             target = labels.cuda(non_blocking=True)
#             optimizer.zero_grad()
#             output = net(images)
#             loss = loss_function(output, target)
#             loss.backward()
#             optimizer.step()
#             output1 = get_mae_age(output)
#             # mae = torch.sum(torch.abs(output - target.unsqueeze(1)), dim=1).mean()
#             # print statistics
#             # running_loss += loss.item()
#             #
#             # train_bar.desc = "train epoch[{}/{}] loss:{:.3f} mae:{:.3f}".format(epoch + 1,
#             #                                                          epochs,
#             #                                                          loss,mae)
#             mae = torch.mean(torch.abs(target - output1)).item()
#             print(output)
#             print("target",target)
#             print("output1",output1)
#             print("mae",mae)
#         # validate
#         # net.eval()
#         # with torch.no_grad():
#         #     val_bar = tqdm(validate_loader, file=sys.stdout)
#         #     for val_data in val_bar:
#         #         val_images, val_labels = val_data
#         #         output = net(val_images.to(device))
#         #         output1 = F.softmax(output, dim=1)
#         #
#         #         # rank = torch.Tensor([i for i in range(101)]).cuda()
#         #         # pred = torch.sum(output * rank, dim=1)
#         #         images = images.cuda(non_blocking=True)
#         #         target = val_labels.cuda(non_blocking=True)
#         #         # loss = loss_function(outputs, test_labels)
#         #         mae = 0
#         #         mae = torch.sum(torch.abs(output1 - target.unsqueeze(1)), dim=1).mean()
#         #
#         #         val_bar.desc = "valid epoch[{}/{}, mae:[{}]]".format(epoch + 1,
#         #                                                    epochs,mae)
#         #
#         #         print(output1.shape)
#         #         print(val_labels.shape)
#         #         print(output.shape)
#         # if mae < best_mae:
#         #     best_mae = mae
#         #     torch.save(net.state_dict(), save_path)
#
#     print('Finished Training')
#
#
# if __name__ == '__main__':
#     main()
