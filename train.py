import argparse
import logging 
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import data_process
#from eval import eval_net
from my_UNet import UNet





""" train_imgs_path = 'data/original_dataset/Train'
train_masks_path = 'data/original_dataset/Train_GT_masks'
train_imgs_save_path = 'data/modified_dataset/Train_resized'
train_masks_save_path = 'data/modified_dataset/Train_GT_resized'

test_imgs_path = 'data/original_dataset/Test'
test_masks_path = 'data/original_dataset/Test_GT_masks'
test_imgs_save_path = 'data/modified_dataset/Test_resized'
test_masks_save_path = 'data/modified_dataset/Test_GT_resized' """

train_imgs_path = 'D:/Users/imbrm/ISIC_2017/Train'
train_masks_path = 'D:/Users/imbrm/ISIC_2017/Train_GT_masks'
train_imgs_save_path = 'D:/Users/imbrm/ISIC_2017/Train_resized'
train_masks_save_path = 'D:/Users/imbrm/ISIC_2017/Train_GT_resized'

test_imgs_path = 'D:/Users/imbrm/ISIC_2017/Test'
test_masks_path = 'D:/Users/imbrm/ISIC_2017/Test_GT_masks'
test_imgs_save_path = 'D:/Users/imbrm/ISIC_2017/Test_resized'
test_masks_save_path = 'D:/Users/imbrm/ISIC_2017/Test_GT_resized'



resize_dimensions = (572,572)
resize_jobs = 10



""" resize_train(train_imgs_path, train_imgs_save_path, size, jobs)
resize_train(train_masks_path, train_masks_save_path, size, jobs)

resize_test(test_imgs_path, test_imgs_save_path, size, jobs)
resize_test(test_masks_path, test_masks_save_path, size, jobs) """



def process_dataset(dataset, size, jobs):
    data_process.del_superpixels('D:/Users/imbrm/ISIC_2017/ay') #TO BE CHANGED 
    data_process.resize_train(train_imgs_path, train_imgs_save_path, resize_dimensions, resize_jobs)
    data_process.resize_train(train_masks_path, train_masks_save_path, resize_dimensions, resize_jobs)



def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp,
              img_scale):
    return

                
            
if __name__ == "__main__":
    data_process.split_train_validation('D:/Users/imbrm/ISIC_2017', "ay.csv", 5)