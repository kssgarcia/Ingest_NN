# %%
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def compute_input(filepath, dest_path, src_path):
    # Read the file and extract the image paths
    with open(filepath, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
        folders = list(set([image.split('/')[0] for image in image_paths]))

    images = []

    # Train part
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for folder in folders:
        if not os.path.exists(os.path.join(dest_path, folder)):
            os.makedirs(os.path.join(dest_path, folder))
    
    for i,image_path in enumerate(image_paths):
        src_image_path = os.path.join(src_path, image_path+".jpg")
        dest_image_path = os.path.join(dest_path, image_path+".jpg")
        shutil.copy(src_image_path, dest_image_path)

    print('Done')
    return images

compute_input('./food-101/meta/test.txt',  './food_101/test/', './food-101/images/')
compute_input('./food-101/meta/train.txt',  './food_101/train/', './food-101/images/')