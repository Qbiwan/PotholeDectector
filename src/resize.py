import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_PATH = "../Dataset/"
OUTPUT_PATH = "../Dataset224/"


def create_output_folder(output_folder):
    '''
    Create ouput-folder for resized images
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Directory ", output_folder, "created")


def resize_image(image_path, output_folder, resize=(224, 224)):
    '''
    Resize images to default size(224,224)
    '''
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    try:
        img = Image.open(image_path)
        img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
        img.save(outpath)
        img.close()
    except (UnidentifiedImageError, OSError):
        pass


def multi_processing_resize(input_folder, output_folder):
    '''
    Enable multi-processing for resizing the images
    '''
    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    Parallel(n_jobs=12)(delayed(resize_image)(image, output_folder, (224, 224))
                        for image in tqdm(images))


if __name__ == "__main__":

    create_output_folder(OUTPUT_PATH + "normal/")
    create_output_folder(OUTPUT_PATH + "potholes/")
    multi_processing_resize(INPUT_PATH+"normal/", OUTPUT_PATH+"normal/")
    multi_processing_resize(INPUT_PATH+"potholes/", OUTPUT_PATH+"potholes/")
    
    if os.path.exists(OUTPUT_PATH+"normal/226.jpg"):        
        os.remove(OUTPUT_PATH+"normal/226.jpg")
        print(f"File {OUTPUT_PATH}normal/226.jpg Removed!") # file 226.jpg is a faulty image file that should be deleted

    if os.path.exists(OUTPUT_PATH+"potholes/312.jpg"):        
        os.remove(OUTPUT_PATH+"potholes/312.jpg")
        print(f"File {OUTPUT_PATH}potholes/312.jpg Removed!") # file 226.jpg is a faulty image file that should be deleted


    normal_images = os.listdir(os.path.join(OUTPUT_PATH+"normal"))
    normal_labels = [0]*len(normal_images)
    normal_images = np.array([("normal/"+img, label) for img, label in zip(normal_images,normal_labels)])

    potholes_images = os.listdir(os.path.join(OUTPUT_PATH+"potholes"))
    potholes_labels = [1]*len(potholes_images)
    potholes_images = np.array([("potholes/"+img, label) for img, label in zip(potholes_images,potholes_labels)])

    all_images = np.vstack((normal_images, potholes_images))
    df = pd.DataFrame(all_images, columns=["image", "label"])
    saved_csv = os.path.join(OUTPUT_PATH + "images_labeled.csv")
    df.to_csv(saved_csv, index=False)
