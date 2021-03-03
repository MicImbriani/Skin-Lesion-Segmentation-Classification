import argparse
import logging 
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data_process
#from eval import eval_net
from my_UNet import UNet
from 



logging.basicConfig(filename='train.log')



resize_dimensions = (572, 572)
resize_jobs = 10


def train_net(
    net,
    device,
    epochs,
    batch_size,
    lr,
    validation_ratio,
    save_cp,
    img_scale
    ):
    #dataset =
    validation_quantity = int(len(dataset) * validation_ratio)
    train_quantity = len(dataset) - validation_quantity

    train, validation = random_split(dataset, [train_quanity, validation_quantity])

    train_loader = DataLoader(train, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    validation_loader = DataLoader(validation,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True,
                                    drop_last=True)

    writer = SummaryWriter(comment=f'LR:{lr}_BS:{batch_size}_VAL:{validation_ratio}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    return

                
            
if __name__ == "__main__":
    data_process.split_train_validation('D:/Users/imbrm/ISIC_2017', "ay.csv", 5)