import os
import sys
import glob
from argparse import ArgumentParser
from model import ModelInterface
from dataset import DataInterface
import utils
import torch
import time

from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger


def load_callbacks(args):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=10,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    # if args.lr_scheduler :
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))

    return callbacks

def main(args):
    # callbacks = load_callbacks(args)
    # logger = TensorBoardLogger(save_dir='apes_log', name=args.filename)
    # print(args)
    
    data_module = DataInterface(**vars(args))
    data_module.setup()
    data_loader = data_module.test_dataloader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModelInterface.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    trainer = Trainer(accelerator='auto')
    trainer.test(model, dataloaders=data_loader)    




if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset 
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset', type=str, default="modelnet_cube")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--less_data', type=bool, default=False)
    parser.add_argument('--one_data', type=bool, default=False) # batch size should be 2
        # train
    parser.add_argument('--pcd_train_path', type=str, default='data/modelnet_cube/pcd/train/')
    parser.add_argument('--cls_label_train_path', type=str, default='data/modelnet_cube/label/train/')
    parser.add_argument('--multi_view_train_path', type=str, default='data/modelnet_cube/pcd/train/alphashape9*9_gradient_img')
    parser.add_argument('--render_train_path', type=str, default='data/modelnet_cube/pcd/train/render_img')
    # parser.add_argument('--pcd_train_path', type=str, default='data/modelnet_small/pcd/train/')
    # parser.add_argument('--cls_label_train_path', type=str, default='data/modelnet_small/label/train/')
    # parser.add_argument('--multi_view_train_path', type=str, default='data/modelnet_small/pcd/train/alphashape9*9_gradient_img')
    # parser.add_argument('--render_train_path', type=str, default='data/modelnet_small/pcd/train/render_img')
    parser.add_argument('--img_savepath', type=str, default='training_imgs')
        # val
    parser.add_argument('--pcd_val_path', type=str, default='data/modelnet_cube/pcd/test/')
    parser.add_argument('--cls_label_val_path', type=str, default='data/modelnet_cube/label/test/')
    parser.add_argument('--multi_view_val_path', type=str, default='data/modelnet_cube/pcd/test/alphashape9*9_gradient_img')
    parser.add_argument('--render_val_path', type=str, default='data/modelnet_cube/pcd/test/render_img')
    # parser.add_argument('--pcd_val_path', type=str, default='data/modelnet_small/pcd/test/')
    # parser.add_argument('--cls_label_val_path', type=str, default='data/modelnet_small/label/test/')
    # parser.add_argument('--multi_view_val_path', type=str, default='data/modelnet_small/pcd/test/alphashape3*3_img')
    # parser.add_argument('--render_val_path', type=str, default='data/modelnet_small/pcd/test/render_img')
    
    # checkpoint
    parser.add_argument('--filename', default='apes-modelnet', type=str)
    parser.add_argument('--checkpoint_path', default='default', type=str)


    args = parser.parse_args()

    # print(args.checkpoint_path)
    main(args)
    
