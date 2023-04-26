#!/root/miniconda3/bin/python
# coding: utf-8

import matplotlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import time


# 热力图相关代码
# 热力图绘制
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# 标注热力图
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def add_confusion_matrix(
        writer,
        cmtx,
        class_names=None,
        global_step=None,
        tag="Confusion Matrix",
):
    """
    writer : 创建的写入tensorboard的写入对象接口
    cmtx ： 目前是传入的一个元组，包含两个混淆矩阵，一个进行了归一化，一个没有
    class_names：类别名，传入 plot_cmtxs，heapmap标注横纵坐标时要用，如果没有，直接标注类12345...
    global_step=None：目前步数，传入writer
    tag="Confusion Matrix"：图名，传入writer
    """

    fig = plot_cmtxs(cmtx, class_names)

    writer.add_figure(tag=tag, figure=fig, global_step=global_step)


class result_storage:
    """
    这是一个结果储存类
    储存：
    （1）保存每次模型的模型结构和名称
    （2）模型训练过程中每个epoch的损失.准确率以及混淆矩阵
    （3）保存每次训练得到的模型参数
    """

    # 构造函数，主要是指定相关模型及文件存贮的文件夹，其余均为默认值
    def __init__(self, model, model_name, file_name):

        """
        保存的训练过程的指标参数格式为
        {"指标1":指标1列表,"指标2":指标2列表...},指标列表长度为总epoch数
        """
        self.model = model  # 这里要保证是浅拷贝过程，同时要保证模型已从GPU转移到了或者复制到了cpu上
        self.model_name = model_name
        self.file_name = file_name

        # 每一步均需要拓展记录的量
        self.loss_dict = {"train": [], "test": []}
        self.acc_dict = {"train": [], "test": []}
        self.cmtx_dict = {"train": [], "test": []}

    # 方便更新每一次epoch对象得到属性值
    def set_values(self, train_index, test_index):

        """
        分别更新训练过程及测试过程一些指标的更新
        输入为训练集和测试集的指标字典（对接训练和测试函数的返回数据）：
        格式为：
        train_index = {"epoch":,"loss":,"accuracy":,"predict":,"labels":}
        test_index = {"epoch":,"loss":,"accuracy":,"predict":,"labels":}
        各指标格式：
        epoch、loss、accuracy均为单个数（int/float）
        predict、labels为均为一个一维数组，且长度相等
        """

        self.epoch = train_index["epoch"]

        # epoch更新，对应训练过程指标更新
        self.train_loss = train_index["loss"]
        self.train_acc = train_index["accuracy"]
        self.train_cmtx = self.calc_cmtx(train_index["labels"], train_index["predict"], nomal="N")

        # 更新所有测试集指标
        self.test_loss = test_index["loss"]
        self.test_acc = test_index["accuracy"]
        self.test_cmtx = self.calc_cmtx(test_index["labels"], test_index["predict"], nomal="N")

        # 每进行一个epoch，全过程指标记录列表更新一个值
        self.extend_values()

    # 记录模型全过程，主要是记录每一个epoch的损失，精确度及混淆矩阵
    def extend_values(self):

        self.loss_dict["train"].append(self.train_loss)
        self.loss_dict["test"].append(self.test_loss)

        self.acc_dict["train"].append(self.train_acc)
        self.acc_dict["test"].append(self.test_acc)

        self.cmtx_dict["train"].append(self.train_cmtx)
        self.cmtx_dict["test"].append(self.test_cmtx)

    # 保存到全过程记录列表到指定文件夹
    def save_tofile(self):

        # 储存相关指标参数
        pd.DataFrame(self.loss_dict).to_csv(self.file_name + r"/loss.csv")
        pd.DataFrame(self.acc_dict).to_csv(self.file_name + r"/accuracy.csv")
        pd.DataFrame(self.cmtx_dict).to_csv(self.file_name + r"/cmtx.csv")

        # 储存模型的结构和参数
        model_savePath = self.file_name + "\model.pkl"
        torch.save(self.model, model_savePath)

        # 保存日志
        log = open(self.file_name + r"/log.txt", mode="w", encoding='utf-8')
        log.write(time.ctime())
        log.write("  %s\n" % self.model_name)
        log.write("epoch:%d\n" % self.epoch)
        log.write("train_accuracy:%f\n" % self.train_acc)
        log.write("test_accuracy:%f\n" % self.test_acc)
        log.close()
        # 储存一张最终结果热力图
        # fig = heatmap_cmtx(self,fig_size = (16,6), class_names= None)

    def calc_cmtx(self, labels, preds, nomal=None):

        """
        labels, preds:数据的真实标签和模型预测标签,传入的数据类型时torch.tensor
        """

        # 计算的混淆矩阵
        # 保证传入数据时先传TRUElabel 再传pred label 保证得到的混淆矩阵行是真实标签 列是预测标签
        if nomal == "N":
            cmtx_unomal = confusion_matrix(labels, preds)
            return cmtx_unomal

        elif nomal == "Y":
            cmtx_nomal = confusion_matrix(labels, preds, normalize="true")
            return cmtx_nomal

        else:
            # 计算的混淆矩阵
            # 保证传入数据时先传TRUElabel 再传pred label 保证得到的混淆矩阵行是真实标签 列是预测标签
            cmtx_unomal = confusion_matrix(
                labels, preds, labels=list(range(0, 10)))

            cmtx_nomal = confusion_matrix(
                labels, preds, labels=list(range(0, 10)), normalize="true")

        return cmtx_unomal, cmtx_nomal

    def heatmap_cmtx(self, fig_size=(16, 6), class_names=None):

        """
        cmtx：传入的混淆矩阵，支持传入多个，目前传入两个，以元组传入
        class_names：类别名称，若不传入则按123456...标注类名
        """

        # 取出元组中的两个恶混淆矩阵
        cmtx_unomal, cmtx_nomal = calc_cmtx(self.labels, self.predict)
        cmtx_unomal = cmtx[0]
        cmtx_nomal = cmtx[1]

        # 确定是否传入了类别名称
        if class_names == None:
            class_names = ["class" + str(i) for i in range(1, cmtx_nomal.shape[0] + 1)]

        # 创建一个画布
        fig = plt.figure(dpi=300, figsize=fig_size)
        ax = fig.subplots(1, 2)

        # 热力图和标注
        if class_names == None:
            class_names = ["class" + str(i) for i in range(1, cmtx_nomal.shape[0] + 1)]

        # 数据未归一化
        im_cmtx_unomal, cbar_unomal = heatmap(cmtx_unomal, row_labels=class_names, col_labels=class_names, ax=ax[0],
                                              cmap="YlGn")
        annotate_heatmap(im_cmtx_unomal, data=cmtx_unomal, valfmt="{x}",
                         textcolors=("black", "white"),
                         threshold=100)
        # 数据归一化
        im_cmtx_nomal, cbar_nomal = heatmap(cmtx_nomal, row_labels=class_names, ax=ax[1], col_labels=class_names,
                                            cmap="YlGn")
        annotate_heatmap(im_cmtx_nomal, data=cmtx_nomal, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=0.05)

        # 设置图像的横纵标签
        ax[0].set_ylabel("True label")
        ax[0].set_xlabel("Predicted label")

        ax[1].set_ylabel("True label")
        ax[1].set_xlabel("Predicted label")

        plt.tight_layout()

        return fig


# 功能测试代码
if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )
    model = model.cuda()
    record = Result_storage(model, "conv_net", r"C:\Users\Administrator\Desktop\test")
    train_index = {"epoch": 1, "loss": 0.05, "accuracy": 0.6, "predict": [1, 2, 3, 4, 1], "labels": [1, 2, 3, 4, 2]}
    test_index = {"epoch": 1, "loss": 0.05, "accuracy": 0.6, "predict": [1, 2, 3, 4, 1], "labels": [1, 2, 3, 4, 2]}
    record.set_values(train_index, test_index)
    record.save_tofile()
