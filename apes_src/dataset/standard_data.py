import torch
from torch import Tensor
import os
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from typing import List
from einops import pack
from PIL import Image
import random

if __name__ == '__main__':
    from transforms import *
else :
    from .transforms import *
# from .transforms import LoadPCD, LoadCLSLabel

class StandardData(data.Dataset):
    METAINFO = dict(classes=('airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                             'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                             'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                             'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                             'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
                             'wardrobe', 'xbox'))

    def __init__(self,
                 metainfo=METAINFO,
                 mode="train",
                 pcd_path='data/modelnet/pcd/train/',
                 cls_label_path='data/modelnet/label/train/',
                 multi_view_path = 'data/modelnet/pcd/train/alphashape_img',
                 render_path = 'data/modelnet/pcd/train/render_img',
                 less_data = False, 
                 one_data = False):
        self.data_path = dict(pcd_path_ = pcd_path, cls_label_path_ = cls_label_path, multi_view_path_ = multi_view_path, render_path_ = render_path)
        self.metainfo = metainfo
        self.mode = mode
        self.file_num = 0
        self.LoadPCD = LoadPCD()
        self.LoadCLSLabel = LoadCLSLabel()
        self.LoadMultiview = LoadMultiview()
        self.LoadRender = LoadRender()
        if self.mode == "train":
            self.ShufflePointsOrder = ShufflePointsOrder()
            self.DataAugmentation = DataAugmentation(axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05)
        self.ToCLSTensor = ToCLSTensor()
        self.PackCLSInputs = PackCLSInputs()
        self.less_data = less_data
        self.data_list = self.load_data_list()
        if less_data:
            num_to_remove = (len(self.data_list) // 10)*9  # 要移除的元素数量
            indices_to_remove = random.sample(range(len(self.data_list)), num_to_remove)  # 随机选择要移除的索引
            self.data_list = [item for i, item in enumerate(self.data_list) if i not in indices_to_remove]  # 创建一个没有这些元素的新列表
        if one_data:
            self.data_list = [self.data_list[1], self.data_list[1], self.data_list[5], self.data_list[5] ]
        self.file_num = len(self.data_list)

    def load_data_list(self):
        data_list = []
        pcd_prefix = self.data_path.get('pcd_path_', None)  # data_prefix.__class__ == dict
        pcd_prefix_list = os.listdir(pcd_prefix)
        pcd_prefix_list.sort()
        cls_label_prefix = self.data_path.get('cls_label_path_', None)
        cls_label_prefix_list = os.listdir(cls_label_prefix)
        cls_label_prefix_list.sort()
        multi_view_prefix = self.data_path.get('multi_view_path_', None)
        multi_view_prefix_list = self.organize_files_into_2d_list(multi_view_prefix)
        render_prefix = self.data_path.get('render_path_', None)
        render_prefix_list = self.organize_files_into_2d_list(render_prefix)

        for pcd_name, cls_label_name, multi_view_names, render_names in zip(pcd_prefix_list, cls_label_prefix_list, multi_view_prefix_list, render_prefix_list):
            data_list.append(dict(classes=self.metainfo['classes'],
                                  pcd_path=os.path.join(pcd_prefix, pcd_name),
                                  cls_label_path=os.path.join(cls_label_prefix, cls_label_name),
                                  multi_view_paths=[os.path.join(multi_view_prefix, name) for name in multi_view_names],
                                  render_paths=[os.path.join(render_prefix, name) for name in render_names]))  
        return data_list

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        data = self.data_list[idx]
        # print('pcd: ', data['pcd_path'])
        # print('multi_view_paths: ', data['multi_view_paths'][0])
        data = self.LoadPCD(data)
        data = self.LoadCLSLabel(data)
        data = self.LoadMultiview(data)
        data = self.LoadRender(data)
        # if self.mode == "train":
            # data = self.ShufflePointsOrder(data)
            # data = self.DataAugmentation(data)
        data = self.ToCLSTensor(data)
        Packed_data = self.PackCLSInputs(data)
        #len(data) = 2 : 'inputs' & 'data_samples'
        labels_onehot =  self.get_gt_cls_labels_onehot(Packed_data['data_samples'])
        return {'inputs': Packed_data['inputs'],
                'data_samples': labels_onehot,
                'multi_view_imgs': data['multi_view_imgs'], # pixel: 0~1
                'multi_view_paths': data['multi_view_paths'], # torch.Size([4, 256, 256, 3])
                'render_imgs': data['render_imgs'], # torch.Size([4, 256, 256])
                'render_paths': data['render_paths']
        }
        
    def get_gt_cls_labels_onehot(self, data_sample: ClsDataSample) -> Tensor:
        labels_list = []
        assert data_sample.gt_cls_label_onehot is not None
        labels_list.append(data_sample.gt_cls_label_onehot)
        labels, ps = pack(labels_list, '* C')  # shape == (B, C)
        return labels

    def organize_files_into_2d_list(self, directory):
        files = os.listdir(directory)
        files.sort()
        organized_files = []
        current_group = []
        current_prefix = None
        for file in files:
            prefix = file.split('_')[0]
            if current_prefix is None or prefix != current_prefix:
                if current_group:
                    organized_files.append(current_group)
                    current_group = []
                current_prefix = prefix
            current_group.append(file)

        if current_group:
            organized_files.append(current_group)
        return organized_files
    
if __name__ == '__main__':
    # 'cd src/dataset'
    # 'python standard_data.py'
    import sys
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../model')
    ))
    from utils import *

    pcd_path ='../../data/modelnet/pcd/train/'
    cls_label_path ='../../data/modelnet/label/train/'
    multi_view_train_path = '../../data/modelnet/pcd/train/alphashape_img'
    multi_view_val_path = '../../data/modelnet/pcd/test/alphashape_img'
    render_path = '../../data/modelnet/pcd/train/render_img'

    dataset = StandardData(pcd_path=pcd_path, 
                        cls_label_path=cls_label_path, 
                        multi_view_path = multi_view_train_path, 
                        render_path = render_path,
                        less_data = False,
                        one_data = False)
    data = dataset[0]

    # print(type(data))
    # print(data['inputs'])
    # print(data['inputs'].shape)
    # print(type(data['data_samples']))
    # print(data['data_samples'])
    print(type(data['multi_view_imgs']))
    print(data['multi_view_imgs'].shape)
    print(type(data['render_imgs']))
    print(data['render_imgs'].shape)

    # # render pcd and save 
    # points = data['inputs'].unsqueeze(0).to(torch.float32).cuda()
    # colors = torch.zeros_like(points).reshape(1,2048,3).cuda()
    # renderer = Renderer('cuda:0', 1)
    # outputs = renderer(points,colors) # (B,3,N)&(B,N,3) -> (B,4,256,256,3)
    # outputs = outputs[0][3].detach().cpu().numpy()
    # outputs = (outputs*255).astype(np.uint8)
    # outputs_img = Image.fromarray(outputs)
    # outputs_img.save(os.path.join('../../dataset_test','test_render.png'))

    # # save gt
    # test_multi_img = data['multi_view_imgs']
    # img = test_multi_img[3].detach().cpu().numpy()
    # img = (img).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save(os.path.join('../../dataset_test','test_gt.png'))