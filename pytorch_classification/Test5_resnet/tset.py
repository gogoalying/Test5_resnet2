from model import resnet34
import torch

import os
import sys
import json
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from model import resnet34
from dataset import TrainM, TestM
from torch.nn import functional as F
from Ascension import Ascension
import torch
from torchvision import datasets, transforms
from torchvision import models


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.load_state_dict(torch.load(r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\pytorch_classification\Test5_resnet\resnet34-333f7ec4.pth"))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        # x = x.view(x.size(0), x.size(1))

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

    net = resnet34()
    # 加载预训练的ResNet-18模型
    resnet = net.resnet34(pretrained=True)

    # 假设你的任务只需要更改最后一层的输出类别数量
    num_classes = 10

    # 获取全连接层之前的特征提取部分（不包括全连接层）
    features = torch.nn.Sequential(*list(resnet.children())[:-1])

    # 创建新的全连接层，将输出类别设置为你的任务所需的数量
    num_filters = resnet.fc.in_features
    classifier = torch.nn.Linear(num_filters, num_classes)

    # 将特征提取部分和新的全连接层组合成新的模型
    model = torch.nn.Sequential(features, classifier)

    # 加载全连接层之前的权重
    pretrained_dict = resnet.state_dict()
    model_dict = model.state_dict()

    # 将预训练模型的权重复制到新的模型中（仅复制相同的键）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

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
            # loss = loss_function(logits, target)
            # loss.backward()
            optimizer.step()
            mae = 0
            # rank = torch.tensor([i for i in range(101)]).cuda()

            output = net(images)
            # individual_tensors = list(output.chunk(output.size(0), dim=0))
            # result_list = []
            # for tensor in individual_tensors:
            #     tensor = tensor.view(-1)
            #     b = Ascension(tensor) # 经过函数扩张到想要的类别
            #     result_list.append(b)
            #     # 将结果保存到一个新的二维张量中
            # output1 = torch.stack(result_list)
            print("原始张量{}\n扩张后张量{}".format(output, mae))

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
