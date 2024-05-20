# %%
import os
import shutil
import yaml
import numpy as np

# Leer el archivo de texto y almacenar los ingredientes en una lista
with open('./datasets/Recipes5k/annotations/ingredients_simplified_Recipes5k.txt', 'r') as file:
    recipes = [linea.strip().split(',') for linea in file.readlines()]

ingredients_list = list(set([i for line in recipes for i in line]))
ingredients_vec = np.array([element for element in ingredients_list if element != ""])

test_input = './datasets/Recipes5k/annotations/test_images.txt'  
val_input = './datasets/Recipes5k/annotations/val_images.txt'  
train_input = './datasets/Recipes5k/annotations/train_images.txt'  

test_output = './datasets/Recipes5k/annotations/test_labels.txt'  
val_output = './datasets/Recipes5k/annotations/val_labels.txt'  
train_output = './datasets/Recipes5k/annotations/train_labels.txt'  

def compute_input(file_path, dest_path, src_path):
    # Read the file and extract the image paths
    with open(file_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for i,image_path in enumerate(image_paths):
        image_path_end = image_path.split('/')[-1]
        src_image_path = os.path.join(src_path, image_path)
        dest_image_path = os.path.join(dest_path, image_path_end)
        shutil.copy(src_image_path, dest_image_path)

def compute_output(labels_path, file_path, dest_path):
    with open(file_path, 'r') as file:
        image_name = [line.strip().split('/')[-1][:-3] for line in file.readlines()]

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    labels = np.loadtxt(labels_path, dtype=int)
    for i, label in enumerate(labels):
        positions = np.where(np.isin(ingredients_vec, recipes[label]))[0]
        np.savetxt(f"{dest_path}/{image_name[i]}txt", positions, fmt="%.i")

def create_yaml(path):
    names_list = ingredients_vec.tolist()

    data = {'names': {i: name for i, name in enumerate(names_list)}}

    # Write the dictionary to a YAML file
    with open(path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


compute_input(val_input, './datasets/Test/val/images',  './datasets/Recipes5k/images')
compute_output(val_output, val_input, './datasets/Test/val/labels')
compute_input(test_input, './datasets/Test/test/images',  './datasets/Recipes5k/images')
compute_output(test_output, test_input, './datasets/Test/test/labels')
compute_input(train_input, './datasets/Test/train/images',  './datasets/Recipes5k/images')
compute_output(train_output, train_input, './datasets/Test/train/labels')
create_yaml('./datasets/Test/data.yaml')
print('Done')

# %%

from ultralytics import YOLO

model = YOLO("yolov8m-cls.pt")

if __name__ == "__main__":
    model.train(project='FoodtoFood', data='./datasets/Recipes5kYOLO', epochs=10, imgsz=256)