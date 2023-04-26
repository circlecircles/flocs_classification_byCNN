#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24/024 10:19
# @Author  : circlecircles
# @FileName: test.py
# @Software: PyCharm

from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# 匹配待拼接的视频
path = r"C:\Users\Administrator\Desktop\c"
# 让文件名按序号排列
print(os.listdir(path))
video_name = sorted([f.split(".")[0] for f in os.listdir(path)], key=lambda x: int(x.split("-")[1]))
print(video_name)

for filename in video_name:
    filename_list = filename.split('-')
    if int(filename_list[1]) != 1 or int(filename_list[0]) <= 14:
        continue
    else:
        concate_list = []
        concate_list.append(filename)
        # 如果有更多的待合并量，更改这里
        for i in range(2, 4):
            filename_list[1] = str(i)
            concate_name = "-".join(filename_list)
            if concate_name in video_name:
                concate_list.append(concate_name)

        print(concate_list)

    # 将同一个concate_list中的视频拼接到一起
    concate_path = [os.path.join(path, name + ".mp4") for name in concate_list]
    concate_video = [VideoFileClip(path_name) for path_name in concate_path]

    # 指定拼接视频储存名称与位置
    final_video = concatenate_videoclips(concate_video)
    final_path = r"C:\Users\Administrator\Desktop\d"
    filename_list.pop(1)
    final_name = "-".join(filename_list)
    print("%s concatenating" % final_name)
    final_video.write_videofile(os.path.join(final_path, final_name + ".mp4"))
