import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def imshow(img, transform):
    img = PIL.Image.open(img)
    if transform is not None:
        img = transform(img)
    img.show()

loader_transform = transforms.CenterCrop(140)
imshow('D:/Users/imbrm/ISIC_2017/ayf/ISIC_0000000.jpg', loader_transform)
