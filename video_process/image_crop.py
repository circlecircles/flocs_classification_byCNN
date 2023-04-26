#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11/011 10:18
# @Author  : circlecircles
# @FileName: image_crop.py
# @Software: PyCharm

import os
from PIL import Image
import cv2


# 批量从图片中扣取同一个像素大小的照片并存贮至一个文件夹中
def crop_pic(file_pathname):
    # 遍历该目录下的所有图片文件
    print(file_pathname)
    i = 0
    for filename in os.listdir(file_pathname):
        print('%s\%s' % (file_pathname, filename))
        i += 1
        img = cv2.imread('%s\%s' % (file_pathname, filename))
        # 裁剪
        cropImg = img[600:900, 800:1100]  # 指定裁剪位置和大小
        cv2.imwrite(r"C:\Users\Administrator\Desktop\data2023\pred\2\%s.jpg" % i, cropImg)
        print('%s已经截取成功' % filename)
        cv2.destroyAllWindows()


crop_pic(r"C:\Users\Administrator\Desktop\data2023\raw\2")
