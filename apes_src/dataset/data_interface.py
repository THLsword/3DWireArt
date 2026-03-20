import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .standard_data import StandardData


class DataInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']

        self.pcd_train_path = kwargs['pcd_train_path']
        self.cls_label_train_path = kwargs['cls_label_train_path']
        self.multi_view_train_path = kwargs['multi_view_train_path']
        self.render_train_path = kwargs['render_train_path']

        self.pcd_val_path = kwargs['pcd_val_path']
        self.cls_label_val_path = kwargs['cls_label_val_path']
        self.multi_view_val_path = kwargs['multi_view_val_path']
        self.render_val_path = kwargs['render_val_path']

        self.less_data = kwargs['less_data']
        self.one_data = kwargs['one_data']
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     self.trainset = self.instancialize(train=True)
        #     self.valset = self.instancialize(train=False)

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.testset = self.instancialize(train=False)

        self.trainset = StandardData(mode="train", 
                                    pcd_path = self.pcd_train_path, 
                                    cls_label_path = self.cls_label_train_path, 
                                    multi_view_path = self.multi_view_train_path, 
                                    render_path = self.render_train_path,
                                    less_data = self.less_data,
                                    one_data = self.one_data)

        self.valset = StandardData(mode="val", 
                                   pcd_path = self.pcd_val_path, 
                                   cls_label_path = self.cls_label_val_path, 
                                   multi_view_path = self.multi_view_val_path, 
                                   render_path = self.render_val_path,
                                   less_data = self.less_data,
                                   one_data = self.one_data)
        self.testset = self.valset

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
