import glob
import matplotlib.pyplot as plt
import math

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

import my_UNet
import MyDataset

dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")


in_train_dir = "D:/Users/imbrm/ISIC_2017-2/Train"
in_train_masks_dir = "D:/Users/imbrm/ISIC_2017-2/Train_GT_masks"

in_val_dir = "D:/Users/imbrm/ISIC_2017-2/Validation"
in_val_masks_dir = "D:/Users/imbrm/ISIC_2017-2/Validation_GT_masks"

batch_size = 64
epochs = 5



""" train_img_ids = [splitext(file)[0]+".jpg" for file in listdir(in_train_dir)]
train_img_masks = [splitext(file)[0]+".png" for file in listdir(in_train_masks_dir)]

train_img_ids.sort()
train_img_masks.sort()



train_images1 = np.zeros((len(train_img_files), 572, 572, 1))
for idx, img_name in enumerate(train_img_files):
    train_images1[idx] = plt.imread(img_name)

train_masks1 = np.zeros((len(train_img_masks), 572, 572, 1))
for idx, mask_name in enumerate(train_img_masks):
    mask = cv2.imread(mask_name)
    ret, thresh_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    training_masks1[idx,:,:,0] = thresh_img[:,:,0]



train_img_ids = []
train_img_masks = []

train_img_ids = [file for file in glob.glob(in_train_dir + "*.png")]
train_img_masks = [ file for file in glob.glob(in_train_masks_dir + "*.png")]

train_img_ids.sort()
train_img_masks.sort()



train_images2 = np.zeros((len(train_img_ids), 572, 572, 1)
for idx, img_name in enumerate(train_img_ids):
    train_images2[idx] = plt.imread(img_name)

train_masks2 = np.zeros((len(train_img_masks), 572,572, 1)
for idx, mask_name in enumerate(train_img_masks):
    mask = cv2.imread(mask_name)
    ret, thresh_img = cv2.threshold(mask, 572, 572, cv2.THRESH_BINARY)
    train_masks2[idx,:,:,0] = thresh_img[:,:,0]

train_images = np.zeros((5000, 572, 572, 1))
train_masks = np.zeros((5000, 572, 572, 1)) """

data preprocess function
dataset = MyDataset("D:/Users/imbrm/ISIC_2017-2")

net = my_UNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    net.train()

    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader: