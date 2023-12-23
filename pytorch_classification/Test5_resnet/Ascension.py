import torch
#
def Ascension(tensor):
    x = tensor
    new_tensor = torch.zeros(10)
    new_tensor[:4] = x[:4]
    new_tensor[5:] = x[-1]
    return new_tensor
# def main():
#     c = torch.tensor([[ 0.4231,  1.1365, -1.0586, -1.8375,  0.3636],
#                         [ 2.3861,  0.4217, -0.7033,  1.2013, -2.9148],
#                         [-1.1491, -0.7201,  0.4488,  0.8229, -0.1599],
#                         [ 3.1377, -0.4423, -1.5852,  0.0971, -1.9763],
#                         [ 0.2321,  0.1271,  0.6531, -0.8673, -0.9006],
#                         [-0.7624, -0.7903,  1.0066,  0.6657,  0.8715],
#                         [-1.2185,  5.7552,  0.0556,  1.4027, -6.2590],
#                         [-0.0851, -0.9202,  0.2091, -1.6077,  1.1220],
#                         [ 1.4197,  0.1701, -1.1608,  3.1498, -4.7502],
#                         [-1.3358, -0.5200,  0.9594, -0.7128,  1.5323],
#                         [-0.1838, -0.7726,  0.2807, -0.6962,  0.5759],
#                         [-1.0499, -1.0920,  0.7018, -0.9025,  1.6699],
#                         [-0.9558, -1.4595,  0.6097, -0.7144,  2.0148],
#                         [-1.2111, -1.4006,  0.9143, -1.4704,  1.6314],
#                         [ 0.2996,  2.9665, -0.9045,  0.7935, -4.7989],
#                         [-0.3315,  2.7911, -0.0824,  0.4196, -2.1041]])
#     individual_tensors = list(c.chunk(c.size(0), dim=0))
#     result_list = []
#     for tensor in individual_tensors:
#         new_tensor = tensor.view(-1)
#         b = Ascension(new_tensor)
#         result_list.append(b)
#     # 将结果保存到一个新的二维张量中
#     new_tensor = torch.stack(result_list)
#     print(new_tensor)
# if __name__ == '__main__':
#         main()

# def Ascension(tensor):
#     x = tensor
#     new_tensor = torch.zeros(62)
#     new_tensor[:33] = x[:33]
#     new_tensor[33:35] = x[33]
#     new_tensor[35] = x[34]
#     new_tensor[36:38] = x[35]
#     new_tensor[38:42] = x[36]
#     new_tensor[42:] = x[37]
#     return new_tensor
# def main():
#     a = torch.arange(38)
#     b = Ascension(a)
#     print("原始张量{}\n扩张后张量{}".format(a, b))
# if __name__ == '__main__':
#         main()


# def rearrange_tensor(input_tensor, output_shape):
#     new_tensor = torch.zeros(output_shape, dtype=input_tensor.dtype)
#     new_tensor[:33] = input_tensor[:33]
#     new_tensor[33:35] = input_tensor[33]
#     new_tensor[35] = input_tensor[34]
#     new_tensor[36:38] = input_tensor[35]
#     new_tensor[38:42] = input_tensor[36]
#     new_tensor[42:] = input_tensor[37]
#     return new_tensor
#
# def main():
#     a = torch.arange(38)
#     b = rearrange_tensor(a, (62,))
#     print(b)
#
# if __name__ == '__main__':
#     main()


# import torch
# import torch.nn as nn
# from torchsummary import summary
#
#
# # 定义VGG网络结构
# class VGG(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(VGG, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#
# # 创建VGG网络实例
# vgg = VGG()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = vgg.to(device)
# # 检查输出层结构和激活函数
# print(vgg.classifier[-1])  # 最后一层全连接层
#
# # 创建一个 GPU 上的随机输入张量
# x = torch.randn(1, 3, 224, 224).to(device)  # 将输入张量移动到 GPU 上
#
# # 使用 summary 函数生成模型的摘要信息，并将其保存为图像
# summary(model, (3, 224, 224))