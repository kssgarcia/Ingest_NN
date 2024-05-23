# %%
import shutil
import os

def move_images(source_directory="./trains/train", target_directory="./trains/val", num_images=1):
    os.makedirs(target_directory, exist_ok=True)
    subdirectories = [name for name in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, name))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(source_directory, subdirectory)
        dest_directory = os.path.join(target_directory, subdirectory)
        os.makedirs(dest_directory, exist_ok=True)
        images = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        for image in images[:num_images]:
            shutil.move(os.path.join(subdirectory_path, image), dest_directory)

move_images()