import os, sys, shutil, csv
import random as rd
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from torchvision import transforms

rootdir = r'D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\data_set\flower_data\flower_photos'#main中使用的，accropped是裁剪后数据集
trainlist = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\data_set\train.csv"
# trainlist = '/ssd2/baozenghao/data/Age/MIVIA/training_caip_contest.csv'
testlist = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\data_set\val.csv"
def loadcsv(data_dir, file):
    imgs = list()
    with open(file, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        next(gt)
        for row in gt: # 逐行读取csv文件
            img_name, age = row[0], row[1]# 赋值name和age
            img_path = os.path.join(data_dir, img_name)
            # data_dir= "/path/to/data"， img_name "image1.jpg"，经过 os.path.join(data_dir, img_name) 操作后，img_path 的值将是 "/path/to/data/image1.jpg"
            age = int(round(float(age)))
            imgs.append((img_path, age))
    return imgs
class TrainM(data.Dataset):#main数据增强
    def __init__(self, transform):
        imgs = loadcsv(rootdir, trainlist)
        random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):#使用train_dataset和dataloader调用每次dataloader会使用一次
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")#打开图像转化成RGB格式

        # label = [normal_sampling(int(age), i) for i in range(101)]#标签分布
        # label = [i if i > 1e-15 else 1e-15 for i in label]
        # label = torch.Tensor(label)#转化tensor方便后续处理

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=10)])#n为数量m为强度

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))#数据增强

        # self.transform.transforms.append(CutoutDefault(20))

        img = self.transform(img)
        return img, age
    def __len__(self):
        return len(self.imgs)

class TestM(data.Dataset):#main
    def __init__(self, transform):
        imgs = loadcsv(rootdir, testlist)
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        # img2 = img.transpose(Image.FLIP_LEFT_RIGHT)向右旋转
        img = self.transform(img)
        # img2 = self.transform(img2)
        return img, age
    def __len__(self):
        return len(self.imgs)

# # 步骤3：设置图像转换操作
# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# # 步骤4：创建TestM实例
# test_dataset = TestM(data_transform["train"])
#
# # 步骤5：访问数据
# image, age= test_dataset[0]
# print(f"First image: {image.shape}, Age: {age}")
#
#
# # 假设test_dataset是你的数据集实例
#
# # 随机选择一个索引
# index = 0
#
# # 获取图像和标签
# image, age= test_dataset[index]
# # 可视化图像
# plt.imshow(image.permute(1, 2, 0))  # 转换为(H, W, C)的顺序
# plt.title(f"Age: {age}")
# plt.axis('off')
# plt.show()