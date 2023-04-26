#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/2/002 16:45
# @Author  : circlecircles
# @FileName: image_label.py
# @Software: PyCharm

import os
import cv2

video_path = r"C:\Users\Administrator\Desktop\concate_video"
video_list = os.listdir(video_path)

i = 0
# 从视屏中逐帧提取照片
for file_name in video_list:

    # 根据分的类别指定储存的文件夹
    name_list = file_name.split('-')
    if float(name_list[1]) <= 1.5:
        save_path = r"C:\Users\Administrator\Desktop\10_4\raw\0"
    elif 2 >= float(name_list[1]) >= 1.5:
        save_path = r"C:\Users\Administrator\Desktop\10_4\raw\1"
    elif 2.5 >= float(name_list[1]) >= 2:
        save_path = r"C:\Users\Administrator\Desktop\10_4\raw\2"
    else:
        save_path = r"C:\Users\Administrator\Desktop\10_4\raw\3"

    cap = cv2.VideoCapture(os.path.join(video_path, file_name))
    frameRate = 1  # 帧数截取间隔（每隔20帧截取一帧）  一秒30帧

    time_start = 800
    frame_start = int(time_start / 1.76)  # 换算成时间800s
    c = frame_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    while (True):
        ret, frame = cap.read()
        seconds = int(1.76 * 2)
        if ret:
            if c % frameRate == 0 and seconds <= 980:
                print("开始截取视频第：" + str(c) + " 帧")
                cv2.imwrite(os.path.join(save_path, str(i) + ".jpg"), frame)  # 这里是将截取的图像保存在本地
                i += 1
            c += 1
            cv2.waitKey(0)
        else:
            print("所有帧都已经保存完成")
            cv2.destroyAllWindows()
            break
    cap.release()

