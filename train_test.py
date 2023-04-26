#!/root/miniconda3/bin/python
# coding: utf-8

import torch
from tqdm import tqdm


# 测试函数
def test(model, test_iter, loss_func, epoch):
    """
    test函数需要输入的参数有：
    模型
    测试集迭代器
    损失函数
    当前epoch
    """
    loss_sum = 0
    acc_sum = 0
    samples = 0

    # 初始化标签和预测值的记录
    preds = []
    labels = []

    for j, (x, y) in enumerate(test_iter):
        # 测试模式
        model.eval()
        labels.append(y)

        # 模型在GPU上进行测试集上的精度测试
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            loss = loss_func(y_hat, y)

            # 记录所有预测标签值
            y_pred = y_hat.argmax(dim=1).cpu()
            preds.append(y_pred)

            # 统计该批量大小
            samples += y.shape[0]

            # 计算本批次loss和acc，计算该批量数据的精度并将其累加,所有的矩阵计算尽量在GPU上完成，这样能加快训练速度
            loss_sum += loss.cpu().item()
            acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

    # 记录该epoch loss及acc
    acc = acc_sum / samples
    los = loss_sum / samples
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)

    # 所有指标以字典的形式返回
    test_index = {"epoch": epoch, "loss": los, "accuracy": acc, "labels": labels, "predict": preds}

    return test_index


# 训练函数
def train(model, loss_func, optimizer, train_iter, epoch, epoch_num):
    """
    模型
    损失函数
    优化器
    训练集迭代器
    当前epoch
    """
    # 初始化oss的记录
    loss_sum = 0
    acc_sum = 0
    samples = 0

    # 初始化标签和预测值的记录
    preds = []
    labels = []

    with tqdm(train_iter, desc='epoch:%d/%d' % (epoch, epoch_num), leave=True, ncols=100, unit='imgs',
              unit_scale=True) as iter_set:
        for j, (x, y) in enumerate(iter_set):
            model.train()
            labels.append(y)
            # 模型在GPU上的训练
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计该批量大小
            samples += y.shape[0]

            # 计算本批次loss
            loss_sum += loss.cpu().item()

            # 计算该批量数据的精度并将其累加,所有的矩阵计算尽量在GPU上完成，这样能加快训练速度
            acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

            # 记录所有预测标签值
            y_pred = y_hat.argmax(dim=1).cpu()
            preds.append(y_pred)
            # 在进度条显示大致的loss与acc
            iter_set.set_postfix(loss=format(loss_sum / samples, ".6f"), acc=format(acc_sum / samples, ".3f"))

    # 计算该epoch loss及acc
    los = loss_sum / samples
    acc = acc_sum / samples
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)

    # 所有指标以字典的形式返回
    train_index = {"epoch": epoch, "loss": los, "accuracy": acc, "labels": labels, "predict": preds}

    return train_index


if __name__ == "main":
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader

    import sys
    sys.path.append("../../personal_packages")
    from result_storage import result_storage
    from train_test import train, test
    from loadImage import split_datapath, flocDataset

    # 从本地加载絮体数据集
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5)])
    train_path, train_label, test_path, test_label = split_datapath(r"../../flocs/pred", 0.2, 0)
    train_data = flocDataset(train_path, train_label, transform=transform)
    test_data = flocDataset(test_path, test_label, transform=transform)

    # 创建数据集迭代器，以便在训练与测试时候按批量大小载入数据
    train_set = DataLoader(train_data, batch_size=256, pin_memory=True, shuffle=True, num_workers=7)
    test_set = DataLoader(test_data, batch_size=256, pin_memory=True, shuffle=True, num_workers=7)

    # 导入模型
    from modified_models import Alex_net, res_net18, modified_inception

    model = Alex_net(3, 3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    # 实例化一结果记录类
    recorder = result_storage(model, "inception9_10", "./result_recorder/inception_910")
    # 开始训练
    epoch = 100
    for i in range(epoch):
        train_index = train(model, loss_func, optimizer, train_set, i + 1, epoch)
        test_index = test(model, train_set, loss_func, i + 1)
        print(test_index["accuracy"])
        recorder.set_values(train_index, test_index)

