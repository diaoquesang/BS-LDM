from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os

class mySingleDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, img_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像

        if self.transform:
            image = self.transform(image)  # 图像预处理
        return image  # 返回图像和标签


class myDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, img_dir, label_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.img_dir = img_dir  # 读取图像路径
        self.label_dir = label_dir  # 读取标签路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        label_path = self.label_dir  # 读取标签文件夹路径
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        # print(file)
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像
        label = cv.imread(os.path.join(label_path, file))  # 用openCV的imread函数读取标签

        if self.transform:
            image = self.transform(image)  # 图像预处理
            label = self.transform(label)  # 标签预处理

        return image, label  # 返回图像和标签
