import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def imshow(img):
    img = PIL.Image.open(img)
    affine = transforms.functional.rgb_to_grayscale(img)
    affine.show()



loader_transform = transforms.CenterCrop(140)
for i in range(5):
    imshow('D:/Users/imbrm/ISIC_2017/ayf/ISIC_0000000.jpg')
