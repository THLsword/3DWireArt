import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
import os
import time
import sys
from einops import repeat, pack, rearrange
from typing import List, Dict
from pytorch_lightning import LightningModule
import numpy as np
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
if __name__ == '__main__':
    from backbones import *
    from heads import *
    from utils import *
    from evaluation import *
    from dataset import *
else :
    from .backbones import *
    from .heads import *
    from .utils import *
    from evaluation import *
    from dataset import *

class ModelInterface(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # network
        self.backbone1 = APESClsBackbone(kwargs['which_ds'])
        self.backbone2 = APESSeg2Backbone(kwargs['which_ds'])
        self.encoder = Encoder()
        # self.head = APESClsHead() # output (B, classes=40)
        # self.seghead = APESSegHead()
        self.new_head = NewOut() # output (B, N)
        # self.new_head2 = NewOut2() # output (B, N)
        # self.MLP_Head = MLP_Head() # output (B, N)
        # self.stdhead = StdHead()
        # self.feature_fusion = Feature_fusion()
        self.renderer = Renderer(kwargs['device'], kwargs['batch_size'])
        self.renderer_t = Renderer(kwargs['device'], kwargs['batch_size'])
        
        # conf
        self.conf = kwargs

        # loss
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.mse_loss = MSELoss(reduction='sum')
        self.acc = Accuracy()
        self.BCE_loss = BinaryCrossEntrophy()
        self.L1 = WeightL1()
        self.Brightness_loss = Brightness_loss
        self.epoch_counter = 0
        self.loss_fn = nn.MSELoss(reduction='mean')

        # lr
        self.lr = kwargs['lr']
        self.lr_scheduler = kwargs['lr_scheduler']

        # dir
        self.filename = kwargs['filename']
        self.img_save_dir = os.path.join(kwargs['save_dir'],self.filename,'vis')
        self.training_img_dir = os.path.join(kwargs['save_dir'],self.filename,'train_img')
        self.val_img_dir = os.path.join(kwargs['save_dir'],self.filename,'val_img')
        self.test_img_dir = os.path.join(kwargs['save_dir'],self.filename,'test_img')
        self.train_img_counter = 0
        self.val_img_counter = 0
        self.test_img_counter = 0

    def forward(self, inputs_pcd: Tensor, inputs_img) -> Tensor:
        # backbone
        # pcd_feature, _, _ = self.backbone1(inputs_pcd) # (B,3,N) -> (B, 3072)
        pcd_feature = self.backbone2(inputs_pcd) # (B,3,N) -> (B, 3, 2048)

        # head-cls
        # pred_cls_logits = self.head(pcd_feature) # (B, 3072) -> (B, 40)

        # new head
        # head_input = rearrange(inputs_pcd,'B C N -> B (C N)')
        # head_input = pcd_feature.unsqueeze(2).repeat(1,1,2048)
        output = self.new_head(pcd_feature) # (B, 3, 2048) -> (B,2048)

        # render
        img_color = output.unsqueeze(2).repeat(1,1,3) # (B,2048) -> (B,N,3)
        true_color = torch.ones_like(img_color)
        outputs = self.renderer(inputs_pcd, img_color) # (B,3,N)&(B,N,3) -> (B,4,256,256,3)
        true_outputs = self.renderer_t(inputs_pcd, true_color) # (B,3,N)&(B,N,3) -> (B,4,256,256,3)

        return None, outputs, true_outputs
            
        
    def training_step(self, batch, batch_idx):
        inputs_pcd = batch['inputs']
        inputs_img = batch['render_imgs']
        GT_imgs    = batch['multi_view_imgs']
        gt_cls_labels_onehot = torch.squeeze(batch['data_samples'])

        pred_cls_logits, rendered_imgs, true_color_imgs = self(inputs_pcd, inputs_img) 

        # save img
        if batch_idx == 0: self.save_training_img(rendered_imgs, GT_imgs, true_color_imgs, 'train')
        
        # losses
        # cls_loss = self.ce_loss(pred_cls_logits, gt_cls_labels_onehot)
        L1_loss = self.L1(rendered_imgs, GT_imgs)
        # Brightness_loss = self.Brightness_loss(rendered_imgs, GT_imgs)
        # losses = cls_loss# + Brightness_loss + L1_loss
        losses = L1_loss

        # self.log('cls_loss', cls_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('L1_loss', L1_loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log('Bright_loss', Brightness_loss, on_step=True, on_epoch=False, prog_bar=True)
        return L1_loss

    def validation_step(self, batch, batch_idx):
        inputs_pcd = batch['inputs']
        inputs_img = batch['render_imgs']
        GT_imgs    = batch['multi_view_imgs']
        gt_cls_labels_onehot = torch.squeeze(batch['data_samples'])

        pred_cls_logits, rendered_imgs, true_color_imgs = self(inputs_pcd, inputs_img) 

        # save img
        if batch_idx == 0: self.save_training_img(rendered_imgs, GT_imgs, true_color_imgs, 'val')
        
        # losses
        # cls_loss = self.ce_loss(pred_cls_logits, gt_cls_labels_onehot)
        L1_loss = self.L1(rendered_imgs, GT_imgs)
        # Brightness_loss = self.Brightness_loss(rendered_imgs, GT_imgs)
        # losses = cls_loss# + Brightness_loss + L1_loss
        losses = L1_loss
        self.log('val_loss', L1_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs_pcd = batch['inputs']
        inputs_img = batch['render_imgs']
        GT_imgs    = batch['multi_view_imgs']
        gt_cls_labels_onehot = torch.squeeze(batch['data_samples'])

        pred_cls_logits, rendered_imgs, true_color_imgs = self(inputs_pcd, inputs_img) 

        # save img
        self.save_test_img(rendered_imgs, GT_imgs, true_color_imgs)
        
        
    # def on_after_backward(self):
    #     # print("t_masked_rendered_imgs.grad: ", self.mse_loss.t_masked_rendered_imgs.grad)
    #     input("Press Enter to continue...")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf['lr'])
        if self.conf['lr_scheduler'] is None:
            return optimizer
        else:
            if self.conf['lr_scheduler'] == 'step':
                scheduler = lrs.StepLR(optimizer, 
                                       step_size=self.conf['lr_decay_steps'], 
                                       gamma=self.conf['lr_decay_rate'])
            elif self.conf['lr_scheduler'] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.conf['lr_decay_steps'],
                                                  eta_min=self.conf['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def loss(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> dict:
        gt_cls_labels_onehot = self.get_gt_cls_labels_onehot(data_samples)
        losses = dict()
        pred_cls_logits = self(inputs)
        ce_loss = self.ce_loss(pred_cls_logits, gt_cls_labels_onehot)
        acc = self.acc.calculate_metrics(pred_cls_logits, gt_cls_labels_onehot)
        losses.update(dict(loss=ce_loss))
        losses.update(dict(acc=acc))
        return losses

    def predict(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> List[ClsDataSample]:
        data_samples_list = []
        pred_cls_logits = self(inputs)
        pred_cls_labels_prob = torch.softmax(pred_cls_logits, dim=1)
        pred_cls_labels = torch.max(pred_cls_labels_prob, dim=1)[1]
        for data_sample, pred_cls_logit, pred_cls_label_prob, pred_cls_label in zip(data_samples, pred_cls_logits, pred_cls_labels_prob, pred_cls_labels):
            data_sample.pred_cls_logit = pred_cls_logit
            data_sample.pred_cls_label_prob = pred_cls_label_prob
            data_sample.pred_cls_label = pred_cls_label
            data_samples_list.append(data_sample)
        return data_samples_list

    def tensor(self, inputs: Tensor, data_samples: List[ClsDataSample]) -> Tensor:
        cls_logits = self(inputs)
        return cls_logits

    @staticmethod
    def get_gt_cls_labels_onehot(data_samples: List[ClsDataSample]) -> Tensor:
        labels_list = []
        for data_sample in data_samples:
            assert data_sample.gt_cls_label_onehot is not None
            labels_list.append(data_sample.gt_cls_label_onehot)
        labels, ps = pack(labels_list, '* C')  # shape == (B, C)
        return labels

    def extract_features(self, inputs: Tensor) -> Tensor:
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def add_image(self, name, pcd: np.ndarray, **kwargs):
        # pcd.shape == (N, C), where C: x y z r g b
        os.makedirs(self.img_save_dir, exist_ok=True)
        saved_path = os.path.join(self.img_save_dir, f'{name}.png')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
        ax.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c=pcd[:, 3:]/255., marker='o', s=2)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)

    def save_training_img(self, img_tensor, GT_imgs, true_color_imgs, mode): # (B,4,img)
        if mode == 'train':
            save_dir = self.training_img_dir
            img_filename = f'{self.train_img_counter}_test_img.png'
            # gt_filename = f'gt_img.png'
            # true_filename = f'true_color_img.png'
            gt_filename = f'{self.train_img_counter}_gt_img.png'
            true_filename = f'{self.train_img_counter}_true_color_img.png'
            self.train_img_counter = self.train_img_counter + 1
        else:
            save_dir = self.val_img_dir
            img_filename = f'test_img{self.val_img_counter}.png'
            gt_filename = f'test_gt.png'
            true_filename = f'test_true_color.png'
            self.val_img_counter = self.val_img_counter + 1

        batch_num = 0
        view_num = random.randint(0, 3)
        os.makedirs(save_dir, exist_ok=True)

        save_interval = 1
        
        if self.train_img_counter % save_interval == 0:
            img = img_tensor[batch_num][view_num]*255
            img_np = img.detach().cpu().numpy()
            img_np = (img_np).astype(np.uint8)
            image = Image.fromarray(img_np)
            image.save(os.path.join(save_dir,img_filename))
  
        if not os.path.exists(os.path.join(save_dir,gt_filename)) and self.train_img_counter % save_interval == 0:
            gt_img = GT_imgs[batch_num]*255
            # gt_img = GT_imgs[batch_num].flip([0])
            gt_img = gt_img[view_num].detach().cpu().numpy()
            gt_img = (gt_img).astype(np.uint8)
            gt_img = Image.fromarray(gt_img)
            gt_img.save(os.path.join(save_dir,gt_filename))       

        if not os.path.exists(os.path.join(save_dir,true_filename)) and self.train_img_counter % save_interval == 0:
            true_color_img = true_color_imgs[batch_num][view_num].detach().cpu().numpy()*255
            true_color_img = (true_color_img).astype(np.uint8)
            true_color_img = Image.fromarray(true_color_img)
            true_color_img.save(os.path.join(save_dir,true_filename))

    def save_test_img(self, img_tensor, GT_imgs, true_color_imgs): # (B,4,img)
        save_dir = self.test_img_dir
        os.makedirs(save_dir, exist_ok=True)
        for batch_num in range(img_tensor.shape[0]):
            for view_num in range(img_tensor.shape[1]):
                img_filename = f'{self.test_img_counter}_test_{view_num}.png'
                img = img_tensor[batch_num][view_num]*255
                img_np = img.detach().cpu().numpy()
                img_np = (img_np).astype(np.uint8)
                image = Image.fromarray(img_np)
                image.save(os.path.join(save_dir,img_filename))
            self.test_img_counter = self.test_img_counter + 1

    def selection_2_color(self, selection):
        mean = selection.mean(dim=1, keepdim=True)
        std = selection.std(dim=1, keepdim=True)
        normalized_tensor = (selection - mean) / std
        colors = 1 / (1 + torch.pow(20, -normalized_tensor))

        values, indices = colors.topk(512, dim=-1) # (B, 2048)->(B, 512)
        # 创建一个全为0的新tensor
        result = torch.ones_like(colors)*0.3

        # 将最大的K个值放置在新tensor的相应位置
        result.scatter_(1, indices, values)

        return result.unsqueeze(2).repeat(1,1,3)



if __name__ == '__main__':
    # 'python src/model/model_interface.py'

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--loss', default='MSEloss', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)

    # dataset 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default="modelnet")
    parser.add_argument('--batch_size', type=int, default=4)
        # train
    parser.add_argument('--pcd_train_path', type=str, default='data/modelnet/pcd/train/')
    parser.add_argument('--cls_label_train_path', type=str, default='data/modelnet/label/train/')
    parser.add_argument('--multi_view_train_path', type=str, default='data/modelnet/pcd/train/alphashape_img')
    parser.add_argument('--img_savepath', type=str, default='training_imgs')
        # val
    parser.add_argument('--pcd_val_path', type=str, default='data/modelnet/pcd/test/')
    parser.add_argument('--cls_label_val_path', type=str, default='data/modelnet/label/test/')
    parser.add_argument('--multi_view_val_path', type=str, default='data/modelnet/pcd/test/alphashape_img')

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], default='cosine', type=str)
    parser.add_argument('--lr_decay_steps', default=190, type=int)
    parser.add_argument('--lr_decay_rate', default=0.6, type=float)
    parser.add_argument('--lr_decay_min_lr', default=0, type=float)

    # train
    parser.add_argument('--which_ds', default="global", type=str) # global / local / new
    parser.add_argument('--max_epochs', default=200, type=int)

    # checkpoint
    parser.add_argument('--save_dir', default='apes_log', type=str)
    parser.add_argument('--filename', default='apes-modelnet', type=str)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    parser.add_argument('--device', default=device, type=str)

    args = parser.parse_args()

    model = ModelInterface(**vars(args)).to(device)

    pcd = torch.rand((args.batch_size,3,2048)).to(device)
    output_img = model.forward(pcd)
    print(output_img.shape)