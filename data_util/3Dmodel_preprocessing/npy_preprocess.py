import os
import numpy as np
import argparse

def main(directory_path):
    # 1. 列出指定路径文件夹下所有的.npy文件
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

    # 2. 使用for循环+numpy读取这些.npy文件
    for file in npy_files:
        file_path = os.path.join(directory_path, file)
        data = np.load(file_path)
        
        # 3. 在指定路径下对每个.npy文件创建相同前缀名的文件夹
        folder_name = file.split('.')[0]  # 去掉文件的扩展名，创建同名文件夹
        folder_path = os.path.join(directory_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # data
        points, normals = data[:, :3], data[:, 3:]
        print(data.shape)
        print(points.shape)
        print(normals.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'tool')
    parser.add_argument('--directory_path', type = str, default = '../吉他/')
    args = parser.parse_args()

    main(args.directory_path)