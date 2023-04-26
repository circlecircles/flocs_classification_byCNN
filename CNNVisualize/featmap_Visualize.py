#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/11/011 17:51
# @Author  : circlecircles
# @FileName: featmap_Visualize.py
# @Software: PyCharm

import torch
import torch.nn as nn


class GuideBackprop:
    """对于一个训练好的模型，选择特定的层使用Guide backprop可视化该层的部分或全部特征图

    属性：
    model:已经训练好的模型
    block: class—name，类名，传入的模型如果有复合block层，则在此处传入block类名，默认为None
    gradient:第一层的输入梯度，用来重构成图片的梯度
    """

    def __init__(self, model, block=None):
        # 一些属性变量，初始化定义值，输出值或重要的中间变量
        self.model = model
        self.block = block
        self.gradient = None
        self.relu_forward_output = []
        # 对传入模型的预处理操作，调模型至eval以及给模型加hook
        self.model.eval()
        self.hook_result()
        self.register_relu_hook()

    # 定义一个hook函数返回输出值，GB算法显示featmap需要反向计算到第一层的值，是模型反向传播的中间变量
    def hook_result(self):
        def hook_function(Module, grad_in, grad_out):
            """主要的作用是对注册了该hook_function的层，获取了该层的中间变量，也可对中间变量进行特定操作改变模型的计算图

            module： nn。module子类，一般是模型的某层
            grad_in: tenor，该层的输入值，在这里即将在反向传播上注册该hook_function因此是梯度
            """
            self.gradient = grad_in[0]

        # 在第一层注册输出结果的hook_function
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    # GB对特征图的可视化关键一点在于对relu激活函数的求导机制不同
    # 在计算时我们需要保留每一个rulu激活函数的激活值；并将其用来将每一个relu激活函数的反向求导所得的梯度值进行小修改（将梯度同样部位激活值小于0的值也置零）
    def register_relu_hook(self):
        """总体上分为两个部分，前向计算保留每一个relu层的激活值；反向传播修改每一次反向传播的梯度值。由3个函数组成。

        relu_forward_hook_function: 保留每一个relu层的激活值
        relu_backward_hook_function: 修改每一次反向传播的梯度值
        最后使用一个遍历循环在传入模型的每一个relu层上注册上述两个hook_function
        """

        def relu_forward_hook_function(Module, tensor_in, tensor_out):
            self.relu_forward_output.append(tensor_out)

        def relu_backward_hook_function(Module, grad_in, grad_out):
            """此处的输入端和输出端，是前向传播时的输入端和输出端，也就是说，上面的output的梯度对应这里的grad_out

            Module: nn.Module类，层，这里是nn.ReLU
            grad_in: tensor,该层对于该层输入值的梯度
            grad_out: tensor,该层对于该层输出值的梯度
            """
            corresponding_forward_output = self.relu_forward_output[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            # print(grad_in[0].size())
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0)
            del self.relu_forward_output[-1]
            return (modified_grad_out,)

        # 将模型中所有的relu层注册前向和反向的钩子
        for pos, Module in self.model.features._modules.items():
            # 先判断模型是否有复合夹层
            if isinstance(Module, self.block):
                for pos_inside, Module_inside in Module._modules.items():
                    if isinstance(Module_inside, nn.ReLU):
                        Module_inside.register_forward_hook(relu_forward_hook_function)
                        Module_inside.register_backward_hook(relu_backward_hook_function)
            else:
                if isinstance(Module, nn.ReLU):
                    Module.register_forward_hook(relu_forward_hook_function)
                    Module.register_backward_hook(relu_backward_hook_function)

    # 对特定层进行GB计算
    def generate_gradient(self, input_image, target_layer, filter_position, is_block=False, inside_layer=None):
        """根据选定的层选择截断进行部分进行前向计算，若是block则进一步在block内部某处进行截断

        input_image： tensor = [B,C,H,W，require_grad = True],模型输入图片，前向计算得特征图
        target_layer: int，目标特征图对应的层， 一般是卷积层,注意该处从第零层开始算
        filter_position: int， 哪一组卷积核，哪一个通道
        is_block： bool，目标是否为一个block复合层，默认为False，若为True一般需要进一步指定指定目标层位置
        inside_layer: int,若为block，进一步指定block内的准确位置，若不是block则不指定
        """
        x = input_image
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

            conv_output = torch.sum(torch.abs(x[0, filter_position]))  # 通过索引取得某个featmap（卷积核多少组，featmap就有多少个）
            conv_output.backward()
            gradients_toArray = self.gradient.data.numpy()[0]
            return gradients_toArray

        else:
            for index, layer in enumerate(self.model.features):
                x = layer(x)
                if index == target_layer:
                    print("targetLayer outputShape:", x.shape)
                    print("targetLayer type:", layer)  # 打印目标层的类名称，帮助检验是否为目标层
                    break

            conv_output = torch.sum(torch.abs(x[0, filter_position]))
            conv_output.backward()
            gradients_toArray = self.gradient.data.numpy()[0]
            return gradients_toArray


if __name__ == "__main__":
    """对功能进行测试
    
    （1）加载一个模型
    （2）加载一张图片并预处理，特别要求的是输入图片tensor inquire_grad = True
    （3）可以获得一个卷积层的不同featmap，也可以获取不同图片的同一个位置的featma
    """
    import modified_models as mymodels
    from mpl_toolkits.axes_grid1 import ImageGrid
    from torchsummary import summary
    from PIL import Image
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")


    def rescale_grads(img, gradtype="all"):
        if gradtype == "pos":
            img = (np.maximum(0, img) / img.max())
        elif gradtype == "neg":
            img = (np.maximum(0, -img) / -img.min())
        else:
            img = img - img.min()
            img /= img.max()
        return img


    model = mymodels.res_net18(3, 2)
    model.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\models\resnet18_cd0513.pkl"), strict=False)

    # 加载图片并进行预处理
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])

    image_dog = Image.open(r"D:\classic_dataset\kagglecatsanddogs_3367a\PetImages\Dog\461.jpg")
    image_cat = Image.open(r"D:\classic_dataset\kagglecatsanddogs_3367a\PetImages\Cat\7720.jpg")
    image_floc = Image.open(r"C:\Users\Administrator\Desktop\floc.png").convert('RGB')

    input_x = transform(image_floc)
    # plt.imshow(input_x.permute(1, 2, 0))
    # plt.show()
    input_x = input_x.view(1, *input_x.size())
    input_x.requires_grad = True

    # 格子显示图片
    fig = plt.figure(figsize=(16, 16), dpi=150)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )

    for ax, i in zip(grid, range(16)):
        GBP = GuideBackprop(model)

        grads = GBP.generate_gradient(input_x, 6, i, is_block=True, inside_layer=3)
        ag = rescale_grads(grads.transpose(1, 2, 0), gradtype="all")
        ax.imshow(ag, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
