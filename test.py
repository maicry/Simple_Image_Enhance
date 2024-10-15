import cv2  # cv2 即opencv的库
import numpy as np  # 给numpy起别名np，该库Numerical Python是python的数学函数库
from PIL import Image
# 最邻近差值 位于ImageEnhance.py文件 第行
# 双线性插值 位于ImageEnhance.py文件 第行


def nearest(image, target_size): # 最邻近差值
    # 1：按照尺寸创建目标图像
    target_image = np.zeros(shape=(target_size[0], target_size[1], 3))
    # 2:计算height和width的缩放因子
    alpha_h = target_size[0] / image.shape[0]
    alpha_w = target_size[1] / image.shape[1]

    for x in range(target_image.shape[0] - 1):
        for y in range(target_image.shape[1] - 1):
            # 3:计算目标图像人任一像素点
            # target_image[tar_x,tar_y]需要从原始图像
            # 的哪个确定的像素点image[src_x, xrc_y]取值
            # 也就是计算坐标的映射关系
            src_x = round(x / alpha_h)
            src_y = round(y / alpha_w)

            # 4：对目标图像的任一像素点赋值
            target_image[x, y] = image[src_x, src_y]

    return target_image


def bilinear(img, out_dim):# 双线性插值实现
    src_h = img.shape[0]
    src_w = img.shape[1]
    dst_h = out_dim[1]
    dst_w = out_dim[0]
    # print("src_h,src_w= ", src_h, src_w)
    # print("dst_h,dst_w= ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 根据几何中心重合找出目标像素的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 找出目标像素最邻近的四个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 代入公式计算
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    result = Image.fromarray(dst_img.astype('uint8')).convert('RGB')
    result.save('output(bilinearest version).png')
    return dst_img


algorithm = input('请输入增强方式（nearest or bilinear?）：')
f = input("请输入改变后的分辨率（split by ,）：")
f_f = f.split(',')
img_x = int(f_f[0])
img_y = int(f_f[1])
if algorithm == 'nearest':
    path = 'test.png'
    image = Image.open(path)
    image = np.array(image)
    target = nearest(image, (img_x, img_y))
    target = Image.fromarray(target.astype('uint8')).convert('RGB')
    target.show('nearest')
    print("Done",end='')

if algorithm == 'bilinear':
    img = cv2.imread("test.png")
    dst = bilinear(img, (img_x, img_y))
    cv2.imshow("blinear", dst)
    print("Done", end='')
    cv2.waitKey()


