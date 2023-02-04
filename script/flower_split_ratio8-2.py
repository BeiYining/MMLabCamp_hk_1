import os
from shutil import copy, rmtree
import random


def remk_dir(dir_path:str) :
    if os.path.exists(dir_path):
        rmtree(dir_path)
    os.makedirs(dir_path)

def remk_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
    os.mknod(file_path)

def main():
    # 保证随机可复现
    random.seed(2023)

    # 将数据集中20%的数据划分到验证集中
    split_ratio = 0.2

    data_root = "/home/yangshuo/past_comp/data/flower/"

    origin_flower_path = os.path.join(data_root, "flower_dataset")
    new_flower_path = os.path.join(data_root, 'work_tmp')

    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    # 获得五个类别
    flower_class = [clz for clz in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, clz))]
    flower_class.sort()

    # 建立保存训练集的文件夹
    train_root = os.path.join(new_flower_path, "train")
    remk_dir(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        remk_dir(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(new_flower_path, "val")
    remk_dir(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        remk_dir(os.path.join(val_root, cla))


##################################################
    for index_cla, cla in enumerate(flower_class):
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)  # 图片名称列表
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_ratio))  # val图片名称列表
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
                with open(new_flower_path + '/' + 'val.txt', mode='a') as f:
                    f.write(str(cla) + '/' + str(image) + ' ' + str(index_cla) + '\n')

            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
                with open(new_flower_path + '/' + 'train.txt', mode='a') as f:
                    f.write(str(cla) + '/' + str(image) + ' ' + str(index_cla) + '\n')

            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


def write_clz() :
    data_root = "/home/yangshuo/past_comp/data/flower/flower_dataset"
    dest = "/home/yangshuo/past_comp/data/flower/work_tmp/classes.txt"
    remk_file(dest)
    for clz in os.listdir(data_root) :
        with open(dest, mode='a') as f:
            f.write(str(clz) + '\n')
    print("All classes were writen")

    


if __name__ == '__main__':
    main()
    write_clz()