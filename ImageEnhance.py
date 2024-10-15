import cv2  # cv2 即opencv的库
import numpy as np  # 给numpy起别名np，该库Numerical Python是python的数学函数库
from PIL import Image
# 最邻近差值 位于ImageEnhance.py文件 第10行
# 双线性插值 位于ImageEnhance.py文件 第31行


def image_enhance(image, output_size, way):
    # 最邻近差值法
    global output_image_1, output_image_2
    if way == 'nearest':
        # 从输入中读取目标分辨率，并创建相应分辨率的图像文件
        output_image_1 = np.zeros(shape=(output_size[0], output_size[1], 3))
        # 计算长宽的缩放比例
        zoom_y = output_size[0] / image.shape[0]
        zoom_x = output_size[1] / image.shape[1]
        # 遍历目标图像的每一个像素点
        for x in range(output_image_1.shape[1] - 1):
            for y in range(output_image_1.shape[0] - 1):
                # 计算出相应的临近点的坐标
                nearest_x = round(x / zoom_x)
                nearest_y = round(y / zoom_y)
                # 将临近点的坐标赋值给output_image对应像素点
                output_image_1[x, y] = image[nearest_x, nearest_y]
        # 转化
        result = Image.fromarray(output_image_1.astype('uint8')).convert('RGB')
        # 储存图片结果
        result.save('output(nearest version).png')

    # 双线性插值法
    if way == 'bilinear':
        # 从输入中读取目标分辨率
        raw_height = image.shape[0]
        raw_width = image.shape[1]
        output_height = output_size[1]
        output_width = output_size[0]
        output_image_2 = np.zeros((output_height, output_width, 3), np.uint8)
        # 计算长宽的缩放比例
        zoom_x = float(raw_width) / output_width
        zoom_y = float(raw_height) / output_height
        for i in range(3):
            for main_y in range(output_height):
                for main_x in range(output_width):
                    # 根据几何中心重合找出目标像素的坐标
                    src_x = (main_x + 0.5) * zoom_x - 0.5
                    src_y = (main_y + 0.5) * zoom_y - 0.5
                    # 找出目标像素最邻近的四个点
                    src_x0 = int(np.floor(src_x))
                    src_x1 = min(src_x0 + 1, raw_width - 1)
                    src_y0 = int(np.floor(src_y))
                    src_y1 = min(src_y0 + 1, raw_height - 1)
                    # 代入公式计算
                    temp0 = (src_x1 - src_x) * image[src_y0, src_x0, i] + (src_x - src_x0) * image[src_y0, src_x1, i]
                    temp1 = (src_x1 - src_x) * image[src_y1, src_x0, i] + (src_x - src_x0) * image[src_y1, src_x1, i]
                    # 赋值给对应的像素点
                    output_image_2[main_y, main_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
        # 转化
        result = Image.fromarray(output_image_2.astype('uint8')).convert('RGB')
        # 储存图片结果
        result.save('output(bilinear version).png')




algorithm = input('请输入增强方式（nearest or bilinear?）：')
f = input("请输入改变后的分辨率（split by ,）：")
f_f = f.split(',')
img_x = int(f_f[0])
img_y = int(f_f[1])
path = 'test.png'
input_image = Image.open(path)
input_image = np.array(input_image)
image_enhance(input_image, (img_x, img_y), algorithm)
print("Done", end='')

