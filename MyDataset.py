from os import listdir
from os.path import splitext
from glob import glob
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

import DataProcess



class MyDataset(Dataset):
    def __init__(self, path, masks_suffix='_segmentation', train_or_val):
        self.train_or_val = train_or_val.capitalize()
        if train_or_val == "Train":
            self.imgs_train_folder = path + "/Train"
            self.masks_train_folder = path + "/Train_GT_masks"
        
        if train_or_val == "Validation":
            self.imgs_val_folder = path + "/Validation"
            self.masks_val_folder = path + "/Validation_GT_masks"

        self.mask_suffix = masks_suffix

        self.train_imgs = [splitext(file)[0] for file in listdir(imgs_train_folder)]
        self.val_imgs = [splitext(file)[0] for file in listdir(imgs_val_folder)]
        self.train_masks = [splitext(file)[0] for file in listdir(masks_train_folder)]
        self.val_masks = [splitext(file)[0] for file in listdir(masks_val_folder)]


        def __len__(self):
            if train_or_val == "Train": 
                return len(self.train_imgs)
            if train_or_val == "Validation":
                return len(self.val_imgs)


        def __getitem__(self, index):
            self.union_imgs = self.train_imgs.append(self.val_imgs)
            self.union_masks = self.train_masks.append(self.val_masks)
            idx = self.union_imgs[index]
            mask_file = glob(self.union_masks + "/" + idx + self.masks_suffix + ".*")
            img_file = glob(self.union_imgs + idx + ".*")

            assert len(mask_file) == 1, \
                f'Either no mask or multiple masks found for ID {idx}: {mask_file}'
            
            assert len(img_file) == 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            
            mask = Image.open(mask_file[0])
            img = Image.open(img_file[0])

            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(img, self.scale)
            mask = self.preprocess(mask, self.scale)

            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            }