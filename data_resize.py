import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed



# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True



# Defining a function for will allow me to parallelise the resizing process.
#
# The following function resizes 1 image at a time. It takes 3 parameters: 
# 1) the current image's path,
# 2) the path of the outpur folder in which resized images will be stored,
# 3) a list containing the width and height to be resised to.
# The following function takes the name (basename) of the current image, 
# creates the path for saving the image in the ouput folder, and then
# opens, resizes and saves the image.

def resize(path, output_dir, resize):
    base_name = os.path.basename(path)
    output_path = os.path.join(output_dir, base_name)
    img = Image.open(path)
    img = img.resize((resize[1], resize[0]), resample=Image,BILINEAR)
    img.save(output_path)



# Resize train images.
# 
# The following code stores the input and output directories, then stores all the 
# names of the images in a list using glob, and executes the resizing in parallel.
# Final size is 1024x1024, in line with the U-Net architecture. 
# For the parallelisation, Parallel and delayed are used. 
# tqdm is used for visual representation of the progress, since the dataset is around
# 30GB, it will take some time to process.

input_folder = "D:\Users\imbrm\ISIC_2020\Dataset\Data"
output_folder = "D:\Users\imbrm\ISIC_2020\Dataset\Resized"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
size = (1024, 1024)
Parallel(n_jobs=10)(delayed(resize)(i, output_folder, size) for i in tqdm(images))



# Resize test images.

input_folder = "D:\Users\imbrm\ISIC_2020\Dataset\Data"
output_folder = "D:\Users\imbrm\ISIC_2020\Dataset\Resized"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
size = (1024, 1024)
Parallel(n_jobs=10)(delayed(resize)(i, output_folder, size) for i in tqdm(images))