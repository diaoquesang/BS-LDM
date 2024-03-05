import random
import os


def traverse_directory(directory):
    # 创建一个空的列表用于存储文件名
    file_names = []
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    # 按照文件名排序（如果你希望的话）
    file_names.sort()
    # 创建一个新的txt文件，并将文件名写入该文件
    with open('dataset.txt', 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


def split_dataset(file_path, train_ratio=0.8):
    # 读取数据集
    with open(file_path, 'r') as f:
        data = f.readlines()
    # 随机打乱数据集
    random.shuffle(data)
    # 计算训练集和测试集的边界
    train_size = int(len(data) * train_ratio)
    # 划分训练集和测试集
    train_set = data[:train_size]
    test_set = data[train_size:]
    # 保存训练集和测试集到对应的txt文件
    with open('train_set.txt', 'w') as f:
        f.writelines(train_set)
    with open('test_set.txt', 'w') as f:
        f.writelines(test_set)
    print(f"数据集已成功划分为训练集和测试集，并保存到对应的txt文件中。")


if __name__ == "__main__":
    # 遍历BS文件夹取出数据
    traverse_directory('BS')
    # 以8：2的比例划分训练集和测试集
    split_dataset('dataset.txt', 0.8)
