import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def imshow(img, single_probability, apply_probability, transform):
    img = PIL.Image.open(img)
    my_transf = transforms.Compose([
        transforms.RandomAffine(degrees=360,
                                scale=(0.4,1.5),
                                shear=[0, 20, 0, 20],
                                fillcolor=0),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomPerspective(),
        transforms.RandomResizedCrop(size=img.size),
        transforms.RandomErasing()])
    list = [transforms.RandomAffine(degrees=360,
                                scale=(0.4,1.5),
                                shear=[0, 20, 0, 20],
                                fillcolor=0),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomPerspective(),
            transforms.RandomResizedCrop(size=img.size)]
            
    affine = transforms.RandomApply(list)

    if transform is not None:
        img = transform(img)
    affine(img).show()



loader_transform = transforms.CenterCrop(140)
for i in range(5):
    imshow('D:/Users/imbrm/ISIC_2017/ayf/ISIC_0000000.jpg', 0.5, 1, None)
