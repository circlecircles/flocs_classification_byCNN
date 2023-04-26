import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


# -----------------------------------------------------------------------------------------------------------------------
# Le-Net
class Le_Net(nn.Module):

    def __init__(self, in_channels, out):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out)
        )

    def forward(self, img):

        y = self.features(img)
        y = y.view(y.shape[0], -1)
        y = self.classifier(y)

        return y


# -----------------------------------------------------------------------------------------------------------------------
# Alex-Net
class Alex_net(nn.Module):

    def __init__(self, in_channels, out):
        super(Alex_net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 128, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, out),
            nn.ReLU()
        )

    def forward(self, x):
        # 特征提取
        y = self.features(x)
        # 全连接分类
        y = self.classifier(y.view(y.shape[0], -1))

        return y


# -----------------------------------------------------------------------------------------------------------------------
# Res-Net18
class residual_block(nn.Module):

    def __init__(self, in_channels, out_channels, s=1, use_1x1conv=True):

        super(residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(s, s), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(s, s))
            self.bn3 = nn.BatchNorm2d(out_channels)

        else:
            self.conv3 = None

        self.relu2 = nn.ReLU()

    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)
            x = self.bn3(x)

        y = self.relu2(y + x)

        return y


class res_net18(nn.Module):

    def __init__(self, in_channels, out):
        super(res_net18, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            residual_block(64, 64, use_1x1conv=False),
            residual_block(64, 64, use_1x1conv=False),
            residual_block(64, 128, s=2, use_1x1conv=True),
            residual_block(128, 128, use_1x1conv=False),
            residual_block(128, 256, s=2, use_1x1conv=True),
            residual_block(256, 256, use_1x1conv=False),
            residual_block(256, 512, s=2, use_1x1conv=True),
            residual_block(512, 512, use_1x1conv=False),

            nn.AvgPool2d(kernel_size=(7, 7), stride=(3, 3))
        )

        self.classifier = nn.Linear(512 * 1 * 1, out)

    def forward(self, x):
        # 首个卷积加池化层
        y = self.features(x)

        # 线性层
        y = self.classifier(y.view(y.shape[0], -1))

        return y


# ----------------------------------------------------------------------------------------------------------------------
# Alex-Net
class Alex_net_raw(nn.Module):

    def __init__(self, in_channels, out):
        super(Alex_net_raw, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, out),
            nn.ReLU()
        )

    def forward(self, x):
        # 特征提取
        y = self.features(x)
        # 全连接分类
        y = self.classifier(y.view(y.shape[0], -1))

        return y

# inception
# 构建inception——block
class inception_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(inception_block, self).__init__()

        # 1x1 conv
        self.conv1_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )

        # （1x1+3x3）conv
        self.conv3_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels * 3 / 8), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels * 3 / 8), int(out_channels / 2), kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU()
        )

        # （1x1+5x5）conv
        self.conv5_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 16), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels / 16), int(out_channels / 8), kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU()
        )

        # （maxpool+1x1）conv
        self.convpool_1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels, int(out_channels / 8), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )

    def forward(self, x):
        y1 = self.conv1_1x1(x)
        y2 = self.conv3_1x1(x)
        y3 = self.conv5_1x1(x)
        y4 = self.convpool_1x1(x)

        y = torch.cat([y1, y2, y3, y4], dim=1)

        return y


# 使用inception——block构建简易版inception网络
class modified_inception(nn.Module):

    def __init__(self, in_channels, out_features):
        super(modified_inception, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception_block1 = inception_block(192, 256)
        self.inception_block2 = inception_block(256, 480)
        self.inception_block3 = inception_block(480, 512)
        self.inception_block4 = inception_block(512, 528)
        self.inception_block5 = inception_block(528, 832)
        self.inception_block6 = inception_block(832, 1024)

        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))

        self.fc_net = nn.Sequential(
            nn.Linear(1 * 1 * 1024, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        y = self.conv_net(x)

        y = self.inception_block1(y)
        y = self.inception_block2(y)
        y = self.pool(y)

        y = self.inception_block3(y)
        y = self.inception_block4(y)
        y = self.inception_block5(y)
        y = self.pool(y)

        y = self.inception_block6(y)
        y = self.avg_pool(y)

        y = self.fc_net(y.view(y.shape[0], -1))

        return y
