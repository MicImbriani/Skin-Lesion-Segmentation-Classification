import os
from os.path import splitext
from os import listdir
import glob
import logging
import random

import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

from torchvision import transforms
import torch


# from dataset import MyDataset


# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True


# Configure logging's details
logging.basicConfig(
    filename="data_process.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def del_superpixels(input_path, jobs):
    """Deletes the superpixels images of the skin lesions.

    Args:
        input_path (string): Path of the folder containing all the images.
        jobs (string): Number of job for parallelisation.
    
    Returns:
        None
    """
    images = [
        splitext(file)[0]
        for file in listdir(input_path)
        if "_superpixels" in splitext(file)[0]
    ]
    print("Deleting Superpixel Images:")
    Parallel(n_jobs=jobs)(delayed(os.remove)(str(input_path + "/" + str(image + ".png"))) for image in tqdm(images))
    logging.info(f"Succesfully deleted {len(images)} SUPERPIXEL images.")


def resize(image, input_folder, output_path, size, image_or_mask):
    """Defining a function that will allow me to parallelise the resizing process.
    The following function resizes 1 image at a time. It takes 3 parameters:
    1) the current image's path,
    2) the path of the output folder in which resized images will be stored,
    3) a (w,h) tuple of the width and height to be resized to.
    The following function takes the name (basename) of the current image,
    creates the path for saving the image in the ouput folder, and then
    opens, resizes and saves the image.

    Args:
        input_path (string): Path to the image.
        output_dir (string): Name of the output directory in which the images will be stored.
        size (tuple): Target size to be resized to.
        image_or_mask (string): States whether it is an image or a mask.

    Returns:
        None
    """
    if image_or_mask == "IMAGE":
        image_path = input_folder + "/" + image + ".jpg"
        img = Image.open(image_path)
        img = img.resize((size[0], size[1]), resample=Image.BILINEAR)
        img.save(output_path + "/" + image + ".jpg")

    if image_or_mask == "MASK":
        image_path = input_folder + "/" + image + ".png"        
        img = Image.open(image_path)
        img = img.resize((size[0], size[1]), resample=Image.BILINEAR)
        img.save(output_path + "/" + image + ".png")


def resize_set(input_folder, output_folder, size, jobs, train_or_test, image_or_mask):
    """
    Stores the input and output directories, then stores all the
    names of the images in a list using glob, and executes the resizing in parallel.
    For the parallelisation, Parallel and delayed are used.
    tqdm is used for visual representation of the progress, since the dataset is around
    30GB, it will take some time to process.

    Args:
        input_folder (string): Path for input folder.
        output_folder (string): Path for output folder.
        size (tuple): Target size to be resized to.
        jobs (int): Number of parallelised jobs.
        train_or_test (string): States whether it's train or test set.
        image_or_mask (string): States whether it is an image or a mask.

    Returns:
        None
    """
    images = [
        splitext(file)[0]
        for file in listdir(input_folder)
    ]
    print(f"Resizing {train_or_test} Images:")
    Parallel(n_jobs=jobs)(delayed(resize)(
        image, input_folder, output_folder, size, image_or_mask
    ) 
        for image in tqdm(images))
    logging.info(f"Resized {len(input_folder)} {train_or_test} images.")


def get_result(image_id, csv_file_path):
    """Checks whether the inputted image was a melanoma or not.

    Args:
        image_id (string): ID of the image.
        csv_file_path (string): Path leading to the .csv file with ground truth.

    Returns:
        melanoma (int): The melanoma classification result in 0 or 1.
    """
    # given image id, finds the row in the csv file and returns results
    df = pd.read_csv(csv_file_path)
    img_in = df.loc[df["image_id"] == image_id]
    img_index = df.loc[df["image_id"] == image_id].index[0]
    melanoma = df.at[img_index, "melanoma"]
    return melanoma


def augment_operations(image_id, image_folder_path, mask_folder_path):
    """Performs augmentation operations on the inputted image.
    Seed is used for using the same randomly generated numbers for applying
    the same augmentation to the image and its mask.

    Args:
        image_id (string): The ID of the image to be augmented.
        image_folder_path (string): Path of folder in which the augmented img will be saved.
        mask_folder_path (string): Path of folder in which the augmented mask will be saved.

    Returns:
        new_img (Image): New augmented PIL Image.
    """
    mask_id = image_id + "_segmentation"
    img = Image.open(image_folder_path + "/" + image_id + ".jpg")
    mask = Image.open(mask_folder_path + "/" + mask_id + ".png")

    transf_comp = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=360, scale=(0.5, 1.1), shear=[0, 20, 0, 20], fillcolor=0
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomPerspective(p=1),
            transforms.RandomResizedCrop(size=img.size),
        ]
    )

    seed = np.random.randint(0, 2 ** 30)
    random.seed(seed)
    torch.manual_seed(seed)
    new_img = transf_comp(img)
    torch.manual_seed(seed)
    new_img_mask = transf_comp(mask)

    return new_img, new_img_mask


def augment_img(image_id, images_folder_path, masks_folder_path, csv_file_path):
    """Executes augmentation on a single image. Due to imbalanced dataset,
    it will perform more augmentation on melanoma images.
    If mole is not melanoma, perform 1 augmentation with probability=0.5.
    If mole is melanoma, perform 4 augmentation with probability=1.
    It performs the same transformation on the image and its relative mask.

    Args:
        image_id (string): ID of the image to be augmented.
        images_folder_path (string): Path of folder in which the augmented img will be saved.
        masks_folder_path (string): Path of folder in which the augmented mask will be saved.
        csv_file_path (string): Path leading to the .csv file with ground truth.

    Raises:
        Exception: [description]
    """
    melanoma = int(get_result(image_id, csv_file_path))
    if melanoma == 0:  # perform augment with 0.5 prob
        augm_probability = 0.5
        n = random.random()
        if n < augm_probability:
            img_1, img_1_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path
            )

            img_1.save(images_folder_path + "/" + image_id + "x1" + ".jpg")
            img_1_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png"
            )

    if melanoma == 1:  # perform 4 augms
        augm_probability = 1

        img_1, img_1_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_2, img_2_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_3, img_3_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_4, img_4_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )

        img_1.save(images_folder_path + "/" + image_id + "x1" + ".jpg")
        img_2.save(images_folder_path + "/" + image_id + "x2" + ".jpg")
        img_3.save(images_folder_path + "/" + image_id + "x3" + ".jpg")
        img_4.save(images_folder_path + "/" + image_id + "x4" + ".jpg")

        img_1_mask.save(
            masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png"
        )
        img_2_mask.save(
            masks_folder_path + "/" + image_id + "_segmentation" + "x2" + ".png"
        )
        img_3_mask.save(
            masks_folder_path + "/" + image_id + "_segmentation" + "x3" + ".png"
        )
        img_4_mask.save(
            masks_folder_path + "/" + image_id + "_segmentation" + "x4" + ".png"
        )


def augment_dataset(images_folder_path, masks_folder_path, csv_file_path, jobs):
    """Performs augmentation on the whole dataset.
    Augmentation is performed in parallel to speed up process.

    Args:
        images_folder_path (string): Path to folder containing images of moles.
        masks_folder_path (string): Path to folder containing images of masks.
        csv_file_path (string): Path to .csv file containing ground truth.
        jobs (int): Number by which the parallelisation will be applied concurrently.

    Returns:
        None
    """
    images = [splitext(file)[0] for file in listdir(images_folder_path)]
    print("Augmenting images:")
    Parallel(n_jobs=jobs)(
        delayed(augment_img)(
            image, images_folder_path, masks_folder_path, csv_file_path
        )
        for image in tqdm(images)
    )
    logging.info(f"Succesfully augmented {len(images)} images.")


def turn_grayscale(image, folder_path):
    """Function for parallelising the grayscale process.

    Args:
        image (string): ID of image to be turn into grayscale.
        folder_path (string): Path leading to folder containing images.
    
    Returns:
        None
    """    
    img = Image.open(folder_path + "/" + image + ".jpg")
    grey = transforms.functional.rgb_to_grayscale(img)
    grey.save(folder_path + "/" + image + ".jpg")



def make_greyscale(folder_path, jobs):
    """Turns all images in a folder from RGB to grayscale.

    Args:
        folder_path (string): Path leading to folder containing images.
        jobs (int): Number of job for parallelisation.
    
    Returns:
        None
    """    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    print("Turning images to GrayScale:")
    Parallel(n_jobs=jobs)(
        delayed(turn_grayscale)(
            image, folder_path
        )
        for image in tqdm(images)
    )
    logging.info(f"Successfully turned {len(images)} images to GrayScale.")


def split_train_validation(input_path, file_name, splits):
    """
    Splits the dataset into train/validation by adding a "k-fold" column in the .csv file and assigning a fold number.
    Shuffles the dataset beforehand (line 3).
    StratifiedKFold because I want cancer/no-cancer ratio to be the same in both sets.

    Args:
        input_path ([string]): [Path of the input directory with the dataset]
        splits ([int]): [Number of cross validation folds to be applied]

    Returns:
        None
    """
    df = pd.read_csv(os.path.join(input_path, file_name))
    df["k-fold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = np.to_df.count()[0]
    kf = model_selection.StratifiedKFold(n_splits=splits)
    for (
        fold,
        train_idx,
        test_idx,
    ) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "k-fold"] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)


def generate_dataset(  #delete,augment,resize,greyscale,split
    train_or_test,
    images_folder_path,
    masks_folder_path,
    output_folder_path,
    masks_output_path,
    csv_file_path,
    resize_dimensions,
    n_jobs
    ):
    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    # Augment with relative masks.
    augment_dataset(
        images_folder_path,
        masks_folder_path,
        csv_file_path,
        n_jobs,
    )

    # Resize images.
    resize_set(
        images_folder_path,
        output_folder_path,
        resize_dimensions,
        n_jobs,
        train_or_test,
        image_or_mask="IMAGE",
    )

    # Resize masks.
    resize_set(
        masks_folder_path,
        masks_output_path,
        resize_dimensions,
        n_jobs,
        train_or_test,
        image_or_mask="MASK",
    )

    # Make images greyscale.
    make_greyscale(
        output_folder_path,
        n_jobs,
    )





if __name__ == "__main__":
    generate_dataset(
        "TRAIN",
        "D:/Users/imbrm/ISIC_2017/check/ayff",
        "D:/Users/imbrm/ISIC_2017/check/ayffs",
        "D:/Users/imbrm/ISIC_2017/check/ayffout",
        "D:/Users/imbrm/ISIC_2017/check/ayffsout",
        "D:/Users/imbrm/ISIC_2017/check/ay.csv",
        (572,572),
        3,
    )