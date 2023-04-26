#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/14/014 7:25
# @Author  : circlecircles
# @FileName: layer_topActivations.py
# @Software: PyCharm
import random

import torch


class LayerActivations:

    def __init__(self, model, image_data, image_path):
        # 初始化成员变量
        self.model = model
        self.image_data = image_data
        self.image_path = image_path  # 列表变量
        self.activations = []
        self.image_label = []
        # 设置成员变量状态
        self.model.eval()

    def calc_activation(self, img, target_layer, filter_position, is_block=False, inside_layer=None):

        x = img
        if is_block:
            for index, layer in enumerate(self.model.features):
                if index != target_layer:
                    x = layer(x)

                elif index == target_layer:
                    for index_inside, layer_inside in enumerate(layer._modules.items()):
                        x = layer_inside[1](x)
                        if index_inside == inside_layer:
                            print("targetLayer outputShape:", x.shape)
                            print("targetLayer type:", layer_inside[1])  # 打印目标层的类名称，帮助检验是否为目标层
                            break  # 结束block内遍历
                    break  # 结束模型的遍历

        else:
            for index, layer in enumerate(self.model.features):
                x = layer(x)
                if index == target_layer:
                    print("targetLayer outputShape:", x.shape)
                    print("targetLayer type:", layer)  # 打印目标层的类名称，帮助检验是否为目标层
                    break

        return torch.sum(torch.sum(torch.abs(x[:, filter_position]), dim=1), dim=1)

    # 对应层激活值冒泡排序
    def bubble_sort(self):
        for i in range(len(self.activations)):
            for j in range(len(self.activations) - i - 1):
                if self.activations[j] < self.activations[j + 1]:
                    # 对激活值进行从大到小排序
                    temp = self.activations[j + 1]
                    self.activations[j + 1] = self.activations[j]
                    self.activations[j] = temp
                    # 对路径进行相同的顺序调整
                    temp = self.image_path[j + 1]
                    self.image_path[j + 1] = self.image_path[j]
                    self.image_path[j] = temp
                    # 对标签进行相应顺序调整
                    temp = self.image_label[j + 1]
                    self.image_label[j + 1] = self.image_label[j]
                    self.image_label[j] = temp

        print("sorted activations:", self.activations)

    def sort_activations(self, target_layer, filter_position, is_block=False, inside_layer=None):
        for index, (img, label) in enumerate(self.image_data):
            x = self.calc_activation(img, target_layer, filter_position, is_block, inside_layer)
            self.activations.append(x)
            self.image_label.append(label)

        # 这里一定要把tensor转成numpy，防止排序出错
        self.activations = torch.cat(self.activations, dim=0).detach().numpy()
        self.image_label = torch.cat(self.image_label, dim=0).detach().numpy()
        print("raw activations:", self.activations)
        # 开始排序
        self.bubble_sort()

        return self.image_path, self.image_label


if __name__ == "__main__":
    from loadImage import flocDataset, split_datapath
    import torchvision
    from torch.utils.data import dataset, DataLoader
    import modified_models as mymodels
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image

    # 验证集样本路径及标签随机采样函数
    def random_index(sequence1, sequence2, sample_num):
        assert len(sequence1) == len(sequence2)
        sample1 = []
        sample2 = []
        random.seed(899)
        sample_index = random.sample(range(0, len(sequence1)), sample_num)
        for index in sample_index:
            sample1.append(sequence1[index])
            sample2.append(sequence2[index])

        return sample1, sample2

    # 加载验证集的全部样本路径及标签
    _, _, test_path, test_label = split_datapath(r"D:\classic_dataset\kagglecatsanddogs_3367a\PetImages", 0.2, 0)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
    )

    # 将从验证集随机抽样出的图片和路径构成一个数据集
    image_path,  image_label = random_index(test_path, test_label, 256)
    image_iter = flocDataset(image_path, image_label, transform=transform)
    image_data = DataLoader(image_iter, batch_size=64, pin_memory=False, shuffle=False)

    # 加载待可视化的模型
    model = mymodels.res_net18(3, 2)
    model.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\models\resnet18_cd0513.pkl"), strict=False)

    # 计算随机从验证集中随机取出的n张图片进行对应层的激活值排序， 返回根据激活值从大到小排序的图片路径和标签
    LA = LayerActivations(model, image_data, image_path)
    path_sort, label_sort = LA.sort_activations(5, 1, is_block=True, inside_layer=3)

    # 格子显示图片
    fig = plt.figure(figsize=(16, 16), dpi=150)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )
    crop_func = torchvision.transforms.Resize((224, 224))
    for ax, path in zip(grid, path_sort[:16]):
        img_read = crop_func(Image.open(path))
        ax.imshow(img_read)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

