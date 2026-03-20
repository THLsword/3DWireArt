import os
import sys
import glob
from argparse import ArgumentParser
from model import ModelInterface
from dataset import DataInterface
import utils
import torch
import time

from pytorch_lightning import Trainer, seed_everything
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
        dirpath=f'{args.save_dir}/{args.filename}/checkpoints',
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


if __name__ == '__main__':
    seed_everything(1234)
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--loss', default='MSEloss', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)

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

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], default='cosine', type=str)
    parser.add_argument('--lr_decay_steps', default=10, type=int)
    parser.add_argument('--lr_decay_rate', default=0.7, type=float)
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

    callbacks = load_callbacks(args)
    logger = TensorBoardLogger(save_dir='apes_log', name=args.filename)
    # print(args)

    trainer = Trainer(
        accelerator='auto',	
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        devices=1,
        benchmark=True, 
        enable_model_summary=True,
    )

    start_time = time.time()

    model = ModelInterface(**vars(args))
    data_module = DataInterface(**vars(args))
    trainer.fit(model, data_module)
    
    end_time = time.time()
    print("Training duration:", end_time - start_time, "seconds")
    # print("PyTorch version:", torch.__version__)
