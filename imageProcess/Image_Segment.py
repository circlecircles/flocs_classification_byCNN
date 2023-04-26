#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/19/019 10:56
# @Author  : circlecircles
# @FileName: Image_segment.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib import cm


def gridFig(image_list, label_list, is_gray, ncols):
    """
    image_list: 列表类型  待展示的图片列表
    label_list: 每张字图片的标题
    is_gray： 布尔类型 是否为灰度图片  影响imshow的参数设置
    """
    nums = len(image_list)
    nrows = int(nums / ncols)
    print(nrows, ncols)
    fig = plt.figure(figsize=(18, 18), dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0,
                     )

    for ax, img, label in zip(grid, image_list, label_list):
        if is_gray:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)

        ax.text(30, 100, label, fontsize=15, c="red")  # 如果输入·图片的尺寸发生变化需要更改此处坐标，以保持文字位置的正确
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(label, loc="left", fontsize=25, pad=-100)
    plt.tight_layout()
    plt.show()


# 三通道彩色图片转灰度图片
def toGray(image):
    """
    image: opencv.imread读入的图片   输入为HWC
    """
    # 单通道法  即R G B各一张
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]

    # 最大值法
    image_max = np.zeros((image_B.shape[0], image_B.shape[1]))
    for i in range(0, image_B.shape[0]):
        for j in range(0, image_B.shape[1]):
            image_max[i][j] = np.max((image[:, :, 0][i][j], image[:, :, 1][i][j], image[:, :, 2][i][j]))
    image_max = image_max.astype(np.uint8)

    # 平均值法
    image_avg = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    image_avg = image_avg.astype(np.uint8)

    # 加权平均法
    image_wht = 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]

    # 将所有方法得到的灰度图加入到最终的结果列表
    image_list = [image_B, image_G, image_R, image_max, image_avg, image_wht]

    return image_list


#
def getGrayHist(image):
    """
    image: cv2.imread读入的图片   输入为HWC
    """
    # 避免中文乱码
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    channel_nums = image.shape[2]
    if channel_nums == 1:
        fig = plt.figure(figsize=(18, 18), dpi=300)
        hist = cv2.calcHist([image[:, :, 0]], [0], None, [256], [0, 256])
        plt.plot(hist.reshape(-1), color=black)
        plt.bar(range(0, 256), hist.reshape(-1), color=black)
        plt.xlim([0, 256])
        plt.xlabel("灰度值", fontsize=20)
        plt.xlabel("灰度值频数", fontsize=20)
        plt.show()
    else:
        fig = plt.figure(figsize=(12, 12), dpi=300)
        channel_names = ["B", "G", "R"]
        for c, c_name, color in zip(range(0, channel_nums), channel_names, ["b", "g", "r"]):
            hist = cv2.calcHist([image[:, :, c]], [0], None, [256], [0, 256])
            plt.plot(hist.reshape(-1), label=c_name, color=color)
            plt.bar(range(0, 256), hist.reshape(-1), color=color)
            plt.xlim([0, 256])

        plt.xlabel("灰度值", fontsize=35, labelpad=0.1)
        plt.ylabel("灰度值频数", fontsize=35)
        plt.tick_params(labelsize=30)
        plt.legend(fontsize=30)

        plt.savefig(r".\3-1.png", dpi=300)


# 图片灰度值随图片空间的分布趋势  用于区分物体像素值与背景像素值的区别
def grayDistrution(gray_image):
    """
    gray_image: 此处输入为单通道的图片， 在三维图中显示多通道灰度的时空分布太乱，意义并不大
    """
    # 首先创建一个3d画布
    fig = plt.figure(figsize=(18, 18), dpi=150)
    ax = fig.add_subplot(projection='3d')
    # 避免中文乱码
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 为了加强对照效果，将原图片显示在底部
    img = gray_image
    # 获取图片长宽
    width = img.shape[0]
    height = img.shape[1]
    # 创建x,y轴的长度
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    z = np.zeros(width * height)
    # 建里颜色列表
    color = img.reshape(-1)
    # 设置颜色，像素
    cmap = plt.get_cmap('gray')
    norm = plt.Normalize(vmin=0, vmax=255)
    # 在z=0的平面上作图
    ax.scatter(x, y, z, c=color, cmap=cmap, norm=norm, alpha=0.5)
    X = np.arange(0, img.shape[1], 1)  # 取出X轴数据
    Y = np.arange(0, img.shape[0], 1)  # 取出Y轴数据
    X, Y = np.meshgrid(X, Y)  # 网格化
    Z = img  # 这里为三维数据

    # 画出灰度分布面
    bins = range(0, 256)
    nbin = len(bins) - 1
    colormap = plt.cm.get_cmap('coolwarm', nbin)
    cNorm = colors.Normalize(vmin=0, vmax=255)
    cBar = cm.ScalarMappable(norm=cNorm, cmap=colormap)
    surf = ax.plot_surface(X, Y, Z, cmap=colormap, norm=cNorm, linewidth=0, antialiased=False)
    cb = fig.colorbar(surf, fraction=0.03)
    # 调整坐标及标注
    plt.xlim((0, 400))
    plt.ylim((0, 400))
    ax.set_zlim((0, 190))
    # 设置主图字体大小
    plt.tick_params(labelsize=30)
    ax.set_xlabel('图片像素横坐标', fontsize=35, labelpad=20)
    ax.set_ylabel('图片像素纵坐标', fontsize=35, labelpad=20)
    ax.set_zlabel('灰度值', fontsize=35, labelpad=20, rotation=90)
    cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小

    plt.tight_layout()
    # plt.show()

    plt.savefig(r".\dis.png", dpi=300)


def grid3d(image_list, label_list):
    nums = len(image_list)

    fig = plt.figure(figsize=(36, 24), dpi=150)
    # 避免中文乱码
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig_num = [231, 232, 233, 234, 235, 236]
    for ax_num, img, label in zip(fig_num, image_list, label_list):
        # 为了加强对照效果，将原图片显示在底部
        ax = fig.add_subplot(ax_num, projection='3d')
        # 获取图片长宽
        width = img.shape[0]
        height = img.shape[1]
        # 创建x,y轴的长度
        x = np.arange(0, width)
        y = np.arange(0, height)
        x, y = np.meshgrid(x, y)
        z = np.zeros(width * height)
        # 建里颜色列表
        color = img.reshape(-1)
        # 设置颜色，像素
        cmap = plt.get_cmap('gray')
        norm = plt.Normalize(vmin=0, vmax=255)
        # 在z=0的平面上作图
        ax.scatter(x, y, z, c=color, cmap=cmap, norm=norm, alpha=0.5)
        X = np.arange(0, img.shape[1], 1)  # 取出X轴数据
        Y = np.arange(0, img.shape[0], 1)  # 取出Y轴数据
        X, Y = np.meshgrid(X, Y)  # 网格化
        Z = img  # 这里为三维数据

        # 画出灰度分布面
        bins = range(0, 256)
        nbin = len(bins) - 1
        colormap = plt.cm.get_cmap('coolwarm', nbin)
        cNorm = colors.Normalize(vmin=0, vmax=255)
        cBar = cm.ScalarMappable(norm=cNorm, cmap=colormap)
        surf = ax.plot_surface(X, Y, Z, cmap=colormap, norm=cNorm, linewidth=0, antialiased=False)
        cb = fig.colorbar(surf, fraction=0.03)
        # 调整坐标及标注
        ax.set_xlim((0, 800))
        ax.set_ylim((0, 800))
        ax.set_zlim((0, 255))
        # 设置主图字体大小

        ax.set_xlabel('图片像素横坐标', fontsize=30, labelpad=15)
        ax.set_ylabel('图片像素纵坐标', fontsize=30, labelpad=15)
        ax.set_zlabel('灰度值', fontsize=30, labelpad=15, rotation=90)
        cb.ax.tick_params(labelsize=30)  # 设置色标刻度字体大小
        ax.tick_params(labelsize=30)

    plt.tight_layout()
    plt.savefig("./cd.png")


# ----------------------------------------------------------------------------------------------------------------------
# 高通低通滤波，写法参照https://blog.csdn.net/xjp_xujiping/article/details/103368158
def combine_images(images, axis=1):
    '''
    合并图像。
    @param images: 图像列表(图像成员的维数必须相同)
    @param axis: 合并方向。
    axis=0时，图像垂直合并;
    axis = 1 时， 图像水平合并。
    @return 合并后的图像
    '''
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
    if axis == 0:  # 垂直方向合并图像
        # 合并图像的 cols
        cols = np.max(shapes[:, 1])
        # 扩展各图像 cols大小，使得 cols一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 垂直方向合并
        return np.vstack(copy_imgs)
    else:  # 水平方向合并图像
        # 合并图像的 rows
        rows = np.max(shapes[:, 0])
        # 扩展各图像rows大小，使得 rows一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 水平方向合并
        return np.hstack(copy_imgs)


def fft(img):
    '''对图像进行傅立叶变换，并返回换位后的频率矩阵'''
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
    # 计算最优尺寸
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    # 根据新尺寸，建立新变换图像
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img
    # 傅立叶变换
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 换位，低频部分移到中间，高频部分移到四周
    return np.fft.fftshift(fft_mat)


def fft_image(fft_mat):
    '''将频率矩阵转换为可视图像'''
    # log函数中加1，避免log(0)出现.
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))
    # 标准化到0~255之间
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))


def ifft(fft_mat):
    '''傅立叶反变换，返回反变换图像'''
    # 反换位，低频部分移到四周，高频部分移到中间
    f_ishift_mat = np.fft.ifftshift(fft_mat)
    # 傅立叶反变换
    img_back = cv2.idft(f_ishift_mat)
    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img_back))
    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))


def fft_distances(m, n):
    '''
    计算m,n矩阵每一点距离中心的距离
    见《数字图像处理MATLAB版.冈萨雷斯》93页
    '''
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1
    # 每点距离矩阵左上角的距离
    ret = np.sqrt(u * u + v * v)
    # 每点距离矩阵中心的距离
    return np.fft.fftshift(ret)


def lpfilter(flag, rows, cols, d0, n):
    '''低通滤波器
    @param flag: 滤波器类型
    0 - 理想低通滤波
    1 - 巴特沃兹低通滤波
    2 - 高斯低通滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹低通滤波的阶数
    @return 滤波器矩阵
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想低通滤波
    if flag == 0:
        filter_mat = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (int(rows / 2), int(cols / 2)),
                   d0, (1, 1, 1), thickness=-1)
    # 巴特沃兹低通滤波
    elif flag == 1:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯低通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat


def hpfilter(flag, rows, cols, d0, n):
    '''高通滤波器
    @param flag: 滤波器类型
    0 - 理想高通滤波
    1 - 巴特沃兹高通滤波
    2 - 高斯高通滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹高通滤波的阶数
    @return 滤波器矩阵
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想高通滤波
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (int(rows / 2), int(cols / 2)),
                   d0, (0, 0, 0), thickness=-1)
    # 巴特沃兹高通滤波
    elif flag == 1:
        duv = fft_distances(rows, cols)
        # duv有 0 值(中心距离中心为0)， 为避免除以0，设中心为 0.000001
        duv[int(rows / 2), int(cols / 2)] = 0.000001
        filter_mat = 1 / (1 + np.power(d0 / duv, 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯高通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat


def do_filter(_=None):
    '''滤波，并显示'''
    d0 = cv2.getTrackbarPos('d0', filter_win)
    flag = cv2.getTrackbarPos('flag', filter_win)
    n = cv2.getTrackbarPos('n', filter_win)
    lh = cv2.getTrackbarPos('lh', filter_win)
    # 滤波器
    filter_mat = None
    if lh == 0:
        filter_mat = lpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    else:
        filter_mat = hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    # 进行滤波
    filtered_mat = filter_mat * fft_mat
    # 反变换
    img_back = ifft(filtered_mat)
    # 显示滤波后的图像和滤波器图像
    cv2.imshow(image_win, combine_images([img_back, fft_image(filter_mat)]))


# -----------------------------------------------------------------------------------------------------------------------
# 图片的傅里叶变换
def fftTrans(gray_image):
    # 读取图像
    img = gray_image

    # 傅里叶变换
    f = np.fft.fft2(img)

    # 转移像素做幅度普
    fshift = np.fft.fftshift(f)

    # 取绝对值：将复数变化成实数取对数的目的为了将数据变化到0-255
    res = np.log(np.abs(fshift))

    # 展示结果
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Fourier Image')
    plt.show()

    return res


# 迭代阈值分割
def Iterate_Thresh(img, initval, MaxIterTimes=20, thre=1):
    mask1, mask2 = (img > initval), (img <= initval)
    T1 = np.sum(mask1 * img) / np.sum(mask1)
    T2 = np.sum(mask2 * img) / np.sum(mask2)
    T = (T1 + T2) / 2
    # 终止条件
    if abs(T - initval) < thre or MaxIterTimes == 0:
        return T
    return Iterate_Thresh(img, T, MaxIterTimes - 1)


def imageSeg(gray_image):
    """
    gray_image: 单通道灰度图片
    """

    # 全局阈值分割算法 全局阈值为150
    _, img_global = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # 自适应阈值分割方法
    img_ada_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,
                                         3)  # 阈值取自相邻区域的平均值
    img_ada_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,
                                             3)  # 阈值取值相邻区域的加权和，权重为一个高斯窗口。

    # 迭代阈值法
    # 计算灰度平均值
    initthre = np.mean(gray_image)
    # 阈值迭代
    thresh = Iterate_Thresh(gray_image, initthre, 50)
    _, img_itrate = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)

    # OTSU
    _, img_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    image_list = [gray_image, img_global, img_ada_mean, img_ada_gaussian, img_itrate, img_otsu]

    return image_list


def openCal(gray_image, kernel_size):
    """
    gray_image:单通道灰度图片
    """
    # 这里的kernel控制腐蚀的效果，kernel的尺寸越大腐蚀效果越严重，同理膨胀也是
    # 控制膨胀和腐蚀的kernel一致保证待识别检测的物体的轮廓不受影响
    erosion = cv2.erode(gray_image, kernel=np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    dilatation = cv2.dilate(erosion, kernel=np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

    return [gray_image, erosion, dilatation]


def contourMark(raw_image, binary_image):
    """
    binary_image: 二值化图片
    """
    floc_contour, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = cv2.drawContours(raw_image, floc_contour, -1, (0, 0, 255), 2)

    # fig = plt.figure(figsize=(18, 18), dpi=150)
    # plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    return cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    image = cv2.imread(r'C:\Users\Administrator\Desktop\data_chapter3\1_pergroup\pred\0.72-14623.jpg', cv2.IMREAD_COLOR)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为RGB
    # getGrayHist(image)

    img_list = toGray(image)
    #image_reverse = 255 - img_list[5]

    # cv2.imwrite(r".\raw.jpg", img_list[5])
    gridFig(img_list, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], True, 3)
    # grayDistrution(image_reverse)
    # grid3d(img_list, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])

    # -----------------------------------------------------------------------------------------------------------------
    """
     # 实现三种高通滤波并显示频域图片及滤波后图片
    img = img_list[5]
    rows, cols = img.shape[:2]
    print(rows,  cols)
    # 滤波器窗口名称
    filter_win = 'Filter Parameters'
    # 图像窗口名称
    image_win = 'Filtered Image'
    cv2.namedWindow(filter_win)
    cv2.namedWindow(image_win)
    # 创建d0 tracker, d0为过滤器大小
    print(min(rows, cols) / 4)
    cv2.createTrackbar('d0', filter_win, 20, VBint(min(rows, cols) / 4), do_filter)
    # 创建flag tracker,
    # flag=0时，为理想滤波
    # flag=1时，为巴特沃兹滤波
    # flag=2时，为高斯滤波
    cv2.createTrackbar('flag', filter_win, 0, 2, do_filter)
    # 创建n tracker
    # n 为巴特沃兹滤波的阶数
    cv2.createTrackbar('n', filter_win, 1, 5, do_filter)
    # 创建lh tracker
    # lh: 滤波器是低通还是高通， 0 为低通， 1为高通
    cv2.createTrackbar('lh', filter_win, 0, 1, do_filter)
    fft_mat = fft(img)
    do_filter()
    cv2.resizeWindow(filter_win, 512, 20)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    # ---------------------------------------------------------------------------------------------------------------
    # # # 实现图像的二值化分割
    # imgseg_list = imageSeg(image_reverse.astype("uint8"))
    # # gridFig(imgseg_list, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], True, 3)
    # l = openCal(imgseg_list[5].astype("uint8"), 7)
    # for im, i in zip(l, range(0,len(l))):
    #     cv2.imwrite(r".\%d.jpg" % i, im)
    # # contourMark(image, imgseg_list[5].astype("uint8"))
    # denoise_list = openCal(imgseg_list[5], 13)
    # denoise_list.append(contourMark(image, denoise_list[2]))
    #
    # label_list = ["(a)", "(b)", "(c)", "(d)"]
    # flag = [0, 0, 0, 1]
    # fig = plt.figure(figsize=(18, 18), dpi=300)
    # grid = ImageGrid(fig, 111,
    #                  nrows_ncols=(2, 2),
    #                  axes_pad=0,
    #                  )
    #
    # for ax, img, label, f in zip(grid, denoise_list, label_list, flag):
    #     if f == 0:
    #         ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    #     else:
    #         ax.imshow(img)
    #
    #     ax.text(30, 100, label, fontsize=15, c="red")  # 如果输入·图片的尺寸发生变化需要更改此处坐标，以保持文字位置的正确
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     # ax.set_title(label, loc="left", fontsize=25, pad=-100)
    # plt.tight_layout()
    # plt.show()
