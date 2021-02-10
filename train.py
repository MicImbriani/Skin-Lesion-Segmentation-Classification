import argparse
import logging 
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from my_U-Net import UNet 




train_imgs_path = 'data/original_dataset/Train'
train_masks_path = 'data/original_dataset/Train_GT_masks'
train_imgs_save_path = 'data/modified_dataset/Train_resized'
train_masks_save_path = 'data/modified_dataset/Train_GT_resized'

test_imgs_path = 'data/original_dataset/Test'
test_masks_path = 'data/original_dataset/Test_GT_masks'
test_imgs_save_path = 'data/modified_dataset/Test_resized'
test_masks_save_path = 'data/modified_dataset/Test_GT_resized'

resize_dimensions = (572,572)
resize_jobs = 10






resize_train(train_imgs_path, train_imgs_save_path, size, jobs)
resize_train(train_masks_path, train_masks_save_path, size, jobs)

resize_test(test_imgs_path, test_imgs_save_path, size, jobs)
resize_test(test_masks_path, test_masks_save_path, size, jobs)