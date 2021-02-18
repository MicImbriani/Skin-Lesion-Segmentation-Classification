import os
import torch 
import U-Net 
import data_process

import numpy as np
import pandas as import pd

import torch.nn as nn 



def run(fold):
    """[summary]

    Args:
        fold (int): Number of k-folds to use for cross validation.
    """    

    train_data_dir = ""



input_folder_train = "D:\Users\imbrm\ISIC_2020\Dataset\Data"
output_folder_train = "D:\Users\imbrm\ISIC_2020\Dataset\Resized"
input_folder_test = "D:\Users\imbrm\ISIC_2020\Dataset\Data"
output_folder_test = "D:\Users\imbrm\ISIC_2020\Dataset\Resized"
size = (572,572)
n_jobs = 10 

data_process.resize_train(input_folder, output_folder, size, n_jobs)
data_process.resize_test(input_folder, output_folder, size, n_jobs)


unet = U-Net()