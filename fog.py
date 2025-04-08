


import os
import cv2
import math
import shutil

def processImage(filepath, destsource):
    '''
    filepath是待处理图片的绝对路径
    destsource是存放雾化后图片的目录
    '''
    # 打开图片
    img = cv2.imread(filepath)
    if img is None:
        print(f"无法读取图片：{filepath}")
        return

    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.02  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心

    for j in range(row):
        for l in range(col):
            d = -0.02 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    # 确保输出目录存在
    if not os.path.exists(destsource):
        os.makedirs(destsource)

    # 保存雾化后的图片
    output_path = os.path.join(destsource, os.path.basename(filepath))
    cv2.imwrite(output_path, img_f * 255)

def processLabels(label_path, dest_labels):
    '''
    复制标签文件到新的目录
    label_path是待处理标签的绝对路径
    dest_labels是存放更新后标签文件的目录
    '''
    if not os.path.exists(dest_labels):
        os.makedirs(dest_labels)

    # 复制标签文件
    shutil.copy(label_path, os.path.join(dest_labels, os.path.basename(label_path)))

def processDataset(image_dir, label_dir, dest_image_dir, dest_label_dir):
    '''
    处理整个数据集，将图像和标签分别存储到指定目录
    image_dir: 原始图片文件夹
    label_dir: 原始标签文件夹
    dest_image_dir: 雾化后图片存储文件夹
    dest_label_dir: 更新标签存储文件夹
    '''
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"未找到标签文件：{label_path}，跳过该图片。")
            continue

        processImage(image_path, dest_image_dir)
        processLabels(label_path, dest_label_dir)

# 示例用法
image_dir = r'D:\w\wjy\da\mydata1_JYZ\images\test'  # 原始图片文件夹路径
label_dir = r'D:\w\wjy\da\mydata1_JYZ\labels\test'  # 原始标签文件夹路径
dest_image_dir = r'D:\w\wjy\da\mydata1_JYZ\images2\test'  # 雾化后图片存储路径
dest_label_dir = r'D:\w\wjy\da\mydata1_JYZ\labels2\test'  # 更新后标签存储路径

processDataset(image_dir, label_dir, dest_image_dir, dest_label_dir)
