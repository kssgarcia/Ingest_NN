# %%
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

model = YOLO("models/USCOMP_bestmodel3.pt")

for param in model.parameters():
    param.requires_grad = True

if __name__ == "__main__":
    add_wandb_callback(model, enable_model_checkpointing=True)
    model.train(project="PlateDetection",data='./datasets/Food_taste/data.yaml', epochs=50, imgsz=640, device=0)
    wandb.finish()

# %%
from ultralytics import YOLO

model = YOLO("../models/bestDataNewonly.pt")

results = model('./badpic.jpg', show=True, conf=0.7)
print(results[0].boxes.xywh)  # print boxes
# %%
from ultralytics import YOLO

model = YOLO("../models/old/USCOMP_bestmodel3.pt")

results = model(source=1, show=True, conf=0.7)


# %%

'''
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
import torch
import os
import torch

wandb.init(project="ultralytics", job_type="inference")
image_dir = './datasets/UASCompVision-1/train/images'
dirs = []
# Iterate over all files in the directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # Check if the file has a .png extension
        if file.endswith('.jpg'):
            # Construct the relative path
            relative_path = os.path.relpath(os.path.join(root, file), image_dir)
            dirs.append(os.path.join(image_dir,relative_path))

model = YOLO("models/USCOMP_bestmodel3.pt")
add_wandb_callback(model, enable_model_checkpointing=True)
predictions = model(dirs)
wandb.finish()

# %%
print(predictions[0].boxes)

'''