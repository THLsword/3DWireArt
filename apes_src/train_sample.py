import os
import sys
import glob

sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))

import logging as log

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
from lib.utils import write_file
from lib.parser import parse_configs
from lib.WireReconstructionSystem import WireReconstructionSystem

seed_everything(1234, workers=True)

# Set logger display format
log.basicConfig(
	format = '[%(asctime)s] [INFO] %(message)s', 
	datefmt = '%d/%m %H:%M:%S',
	level = log.INFO
)

torch.cuda.empty_cache()

def save_config(args, exp_name, conf_text):
    conf_name = args.conf_path.split("/")[-1]
    save_path = f"exp/{exp_name}"
    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, conf_name)
    write_file(save_path, conf_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Train patch/curve representation for 3D point clouds.'
    )
    parser.add_argument('--conf-path', type=str, default='configs/animal.conf')
    args = parser.parse_args()
    conf, conf_text = parse_configs(args.conf_path)
    
    # Get training parameters from the config file.
    exp_name = conf.general.exp_name
    train_kwargs = {
        "n_epochs": conf.get_int("train.n_epochs"),
        "n_val_epochs": conf.get_int("train.n_val_epochs"),
        "save_every": conf.get_int("train.save_every"),
    }

    # Save the config file in `exp/(exp_name)`
    save_config(args, exp_name, conf_text)

    # Set the model checkpoint and logger
    # For more information: https://www.pytorchlightning.ai/
    ckpt_cb = ModelCheckpoint(
        dirpath=f'exp/{exp_name}/checkpoints',
        filename='{epoch:d}',
        every_n_train_steps=train_kwargs["save_every"],
        save_top_k=5,
        monitor='val/loss',
        mode='min',
        save_last=True,
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]
    logger = TensorBoardLogger(
        save_dir='exp',
        name=exp_name,
    )

    # Initialize the trainer and training system
    resume_checkpoint = None
    ckpt = sorted(glob.glob(os.path.join(f"exp/{exp_name}/checkpoints", "*.ckpt")))
    if (ckpt):
        log.info(f"Resume training on: {ckpt[-1]}")
        resume_checkpoint = ckpt[-1]

    system = WireReconstructionSystem(conf)
    trainer = Trainer(	
        max_epochs=train_kwargs["n_epochs"],
        check_val_every_n_epoch=train_kwargs["n_val_epochs"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=1,
        benchmark=True, 
    )

    # Train
    # trainer.fit(system, ckpt_path=resume_checkpoint) 
    trainer.fit(system, ckpt_path=None) 

