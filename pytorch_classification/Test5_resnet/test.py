import torch

def get_mae_age(a):
    a_list = torch.split(a, 1, dim=0)
    array_of_tensors = list(a_list)
    result_list = []

    for i in array_of_tensors:
        max_col_index = torch.argmax(i, dim=1)
        predict_age = max_col_index.item()
        result_list.append(predict_age)
    result_tensor = torch.tensor(result_list, dtype=torch.float32).cuda()
    return result_tensor
# def main():
#     a = torch.tensor([[ 0.4231,  1.1365, -1.0586, -1.8375,  0.3636],
#             [ 2.3861,  0.4217, -0.7033,  1.2013, -2.9148],
#             [-1.1491, -0.7201,  0.4488,  0.8229, -0.1599],
#             [ 3.1377, -0.4423, -1.5852,  0.0971, -1.9763],
#             [ 0.2321,  0.1271,  0.6531, -0.8673, -0.9006],])
#     get_mae_age(a)
# if __name__ == '__main__':
#     main()
# for tensor in individual_tensors:
#      tensor = tensor.view(-1)
#      b = Ascension(tensor) # 经过函数扩张到想要的类别
#      result_list.append(b)
#     # 将结果保存到一个新的二维张量中
