import os
import csv

data_dir = r"D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\data_set\flower_data\val"
csv_file = r'D:\LenovoQMDownload\github\网络\deep-learning-for-image-processing\data_set\val.csv'

# 获取文件夹中的所有文件夹名
folders = os.listdir(data_dir)
# 创建 CSV 文件并写入标题行
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    # 遍历文件夹中的所有文件夹
    for folder_index, folder in enumerate(folders, start=0):  # start=0确保标签从0开始
        folder_path = os.path.join(data_dir, folder)
        files = os.listdir(folder_path)
        # 遍历文件夹中的所有文件
        for file in files:
            # 如果是图片文件
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                # 将文件路径和标签写入 CSV 文件中
                path = os.path.join(folder_path, file)
                label = folder_index  # 使用文件夹索引作为标签
                writer.writerow([path, label])
# # 获取文件夹中的所有文件名
# files = os.listdir(data_dir)
#
# # 创建 CSV 文件并写入标题行
# with open(csv_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['path', 'label'])#可不写
#
#     # 遍历文件夹中的所有文件
#     for file in files:
#
#         # 如果是图片文件
#         # 使用 .lower() 方法将文件名转换为小写，我们可以忽略扩展名的大小写，并同时处理 .jpg 和 .JPG 文件
#         if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
#             # 将文件路径和标签写入 CSV 文件中
#             path = os.path.join(data_dir, file)
#             label = os.path.splitext(file)[0][-2:]  # 使用文件名作为标签
#             writer.writerow([path, label])
