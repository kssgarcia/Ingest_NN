# %%
import os
import shutil
from pathlib import Path

# Define the paths
base_path = Path('../datasets/Food-5K')
train_path = base_path / 'train'
val_path = base_path / 'val'

# Define the target paths
target_train_path = base_path / 'organized_train'
target_val_path = base_path / 'organized_val'

# Create target directories
for target_path in [target_train_path, target_val_path]:
    (target_path / 'food').mkdir(parents=True, exist_ok=True)
    (target_path / 'non_food').mkdir(parents=True, exist_ok=True)

# Function to move files based on their names
def organize_files(source_path, target_path):
    for file_name in os.listdir(source_path):
        if file_name.startswith('0_'):
            shutil.move(source_path / file_name, target_path / 'non_food' / file_name)
        elif file_name.startswith('1_'):
            shutil.move(source_path / file_name, target_path / 'food' / file_name)

# Organize train and validation sets
organize_files(train_path, target_train_path)
organize_files(val_path, target_val_path)

print("Dataset organized successfully!")
