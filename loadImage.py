import os
import json
import random
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from PIL import Image

"""
使用说明：
split_datapath函数：
（1）的主要作用是划分出训练集和测试集路径列表和标签列表并形成对应关系（即路径列表文件和标签列表标签对应）
（2）其需要传入的参数为一个根目录、测试集比例和一个随机数种子。
（3）数据集存储要求，根目录下每一类数据样本单独存放于一个文件夹，类名及文件夹名（不允许出现不存储数据样本的文件夹）
（4）其他功能：会输出一个类名与标签的对应关系以及各类类样本数量的统计
flocDataset类：
（1）主要功能是根据所给文件路径与标签，读取对应的文件并匹配标签，支持批量
（2）对方法进行改写可实现多标签等特殊功能
"""


def split_datapath(root,validate_rate,randomseed):

    # 先确定当前root文件夹下有几个子文件夹，有几个子文件夹，即分成几类
    class_name = [f_name for f_name in os.listdir(root) if os.path.isdir(os.path.join(root,f_name))]
    class_name.sort()

    # 按照文件夹排列顺序，给每个文件夹分配一个label，按顺序0,1,2...{文件名：标签}
    label_dict = dict((c_name,c_label) for c_label,c_name in enumerate(class_name))

    # 初始化训练集和测试集的图片路径及标签列表,以及训练集和测试集中各类别的样本数量列表
    train_path = []
    train_label = []
    test_path = []
    test_label = []
    num_clist = []
    # 总有效文件计数器
    path_num = 0

    # 开始逐变量根目录下的子文件夹并遍历子文件夹下的各图片文件
    for c_name in class_name:

        # 类别计数器，统计该类样本数量
        class_num = 0

        #先拼接图片文件夹路径,并确定读取文件夹中何种格式的文件（我们需要读取图片，一般就是下面四种格式中的一种）
        #如果是三层文件夹从这里开始修改，再获取一次所有二级子文件夹名，加一个循环遍历所有二级子文件夹
        class_path = os.path.join(root,c_name)
        target_file = [".jpg", ".JPG", ".png", ".PNG"]
        #获取该子文件夹下所有目标后缀文件的文件路径,并将该文件夹下的总有效文件数计入总数
        image_path = [os.path.join(root,c_name,i) for i in os.listdir(class_path)
                      if os.path.splitext(i)[-1] in target_file]
        path_num += len(image_path)
        class_num += len(image_path)

        # 设定随机数种子，并随机采样该文件夹下的部分图片为验证集
        random.seed(randomseed)
        validate_image = random.sample(image_path,  k = int(len(image_path)*validate_rate))

        # 遍历该子文件夹图所有的图片路径并将路径根据测试集和训练集加入到对应的列表中
        class_label = label_dict[c_name]
        for path in image_path:
            if path not in validate_image:
                train_path.append(path)
                train_label.append(class_label)
            else:
                test_path.append(path)
                test_label.append(class_label)

        # 记录该类的样本总数
        num_clist.append(class_num)

    # 确保改文件夹下总有效文件数等于训练集和测试集样本总数和
    assert path_num == len(train_label) +  len(test_label)

    #产生一个json文件记录类别与标签的关系，以及每个类的样本数量（训练集+测试集）
    json_label = json.dumps(dict((key,val) for key, val in label_dict.items()), indent=4)
    json_num = json.dumps(dict((key,val) for key, val in zip(label_dict.keys(),num_clist)), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_label)
        json_file.write(json_num)

    return  train_path,train_label,test_path,test_label


# 定义加载絮体数据集的类，继承自dataset类
class flocDataset(Dataset):

    # 构造方法，定义传入的读取路径，及标签
    def __init__(self,data_path,data_label,transform = None):
        super(flocDataset,self).__init__()
        self.data_path = data_path
        self.data_label = data_label
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    # 样本数据及标签的索引读取
    def __getitem__(self, item):

        # 先读取图片，为保证在torch模型中能正常输入，应保证读取图片应为[C,W,H]形状的矩阵
        data = Image.open(self.data_path[item])
        if  data.mode != 'RGB':
            data = data.convert('RGB')
            #raise ValueError("image: {} isn't RGB.".format(self.data_path[item]))
        label = self.data_label[item]

        if self.transform != None:
            data = self.transform(data)

        return data, label

    # 批量打包数据，保证能一次以[B,C,W,H]格式输出一组样本
    # 同时改写 __getitem__和collate_fn就可以实现多标签等特殊功能
    @staticmethod
    def collate_fn(batch):
        datas,labels = tuple(zip(*batch))

        datas = torch.stack(datas,dims=0)
        labels = torch.as_tensor(labels)

        return datas, labels

if __name__ == "__main__":
    transform = torchvision.transforms.ToTensor()
    train_path,train_label,_,_ = split_datapath(r"D:\桌面资料\藻类絮体机器视觉\实验方案与结果记录\结果记录\211126\test_set",0.2,0)
    train_data = flocDataset(train_path,train_label,transform=transform)
    train_set = DataLoader(train_data, batch_size=64, pin_memory=False, shuffle=True)
    iter_set = iter(train_set)
    exp,label = iter_set.__next__()
    print(exp.shape)
    print(label.shape)