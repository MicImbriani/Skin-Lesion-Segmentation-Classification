import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from myTransforms import RandomVerticalFlip, RandomPerspective, ColorJitter

def imshow(img, transform):
    img = PIL.Image.open(img)
    flip = RandomVerticalFlip(1.0)
    persp = RandomPerspective()
    jitt = ColorJitter()
    if transform is not None:
        img = transform(img)
    jitt(img).show()

loader_transform = transforms.CenterCrop(140)
for i in range(5):
    imshow('D:/Users/imbrm/ISIC_2017/ayf/ISIC_0000000.jpg', None)
