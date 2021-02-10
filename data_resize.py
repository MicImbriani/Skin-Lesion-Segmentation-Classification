import os
import glob
import logging

import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

from dataset import MyDataset



# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True


# Configure logging's details
logging.basicConfig(filename='data_resize.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')



def resize(input_path, output_dir, size):
    """Defining a function that will allow me to parallelise the resizing process.
    The following function resizes 1 image at a time. It takes 3 parameters: 
    1) the current image's path,
    2) the path of the output folder in which resized images will be stored,
    3) a list containing the width and height to be resised to.
    The following function takes the name (basename) of the current image, 
    creates the path for saving the image in the ouput folder, and then
    opens, resizes and saves the image.

    Args:
        input_path ([string]): [Path to the image]
        output_dir ([string]): [Name of the output directory in which the images will be stored]
        size ([tuple]): [Target size to be resized to]

    Returns:
        None
    """    
    image_id = os.path.basename(input_path)
    output_path = os.path.join(output_dir, image_id)

    img = Image.open(input_path)
    img = img.resize((size[1], size[0]), resample=Image.BILINEAR)
    img.save(output_path)


def resize_train(input_folder, output_folder, size, jobs):
    """
    Stores the input and output directories, then stores all the 
    names of the images in a list using glob, and executes the resizing in parallel.
    For the parallelisation, Parallel and delayed are used. 
    tqdm is used for visual representation of the progress, since the dataset is around
    30GB, it will take some time to process.

    Args:
        input_folder ([string]): [Path for input folder]
        output_folder ([string]): [Path for output folder]
        size ([tuple]): [Target size to be resized to]
        jobs ([int]): [Number of parallelised jobs]

    Returns:
        None
    """    
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=jobs)(delayed(resize)(i, output_folder, size) for i in tqdm(images))
    logging.info(f'Resized {len(input_folder)} TRAIN images.')


def resize_test(input_folder, output_folder, size, jobs):
    """
    Same as resize_train but for the test set.

    Args:
        input_folder ([string]): [Path for input folder]
        output_folder ([string]): [Path for output folder]
        size ([tuple]): [Target size to be resized to]
        jobs ([int]): [Number of parallelised jobs]

    Returns:
        None
    """    
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=jobs)(delayed(resize)(i, output_folder, size) for i in tqdm(images))
    logging.info(f'Resized {len(input_folder)} TEST images.')


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
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=splits)
    for fold, train_idx, test_idx, in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "k-fold"] = fold
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)


if __name__ == "__main__":
    
    