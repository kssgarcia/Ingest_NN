# %%
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

def get_images_paths(path): 
    return list(path.rglob('*.jpg')) + list(path.rglob('*.png'))

# Load your data
data = pd.read_csv("dataset/full_dataset.csv")
directory_path = "./testRecipes"

labels_to_images = {name: get_images_paths(Path(f"{directory_path}/train/"+name).resolve()) for name in os.listdir(f"{directory_path}/train") if os.path.isdir(os.path.join(f"{directory_path}/train", name))}

# Flatten the dataset for DataLoader
images_paths = []
descriptions = []

for title, paths in labels_to_images.items():
    # matching = data[data['title'] == title]
    # if matching.empty: continue
    for path in paths:
        # desc = f"A food called {title}. Made with {matching['NER'].values[0]}. With a recipe {matching['ingredients'].values[0]}"
        desc = f"A food called {title}"
        images_paths.append(path)
        descriptions.append(desc.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("-", ""))

# Setup model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from transformers import CLIPProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from transformers import AutoProcessor, AutoModel
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Load the model state dictionary
model_path = "./trains/siglip_model_1.pth"  # Replace with the correct path to your saved model
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)

# Freeze all parameters except for the last few layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the last transformer block
for param in model.text_model.encoder.layers[-2:].parameters():
    param.requires_grad = True

for param in model.vision_model.encoder.layers[-2:].parameters():
    param.requires_grad = True

# Ensure the model is in evaluation mode
model.eval()
# %%
# Initialize lists to collect predictions and ground truths
all_preds = []
all_labels = []
descriptions_clean = list(set(descriptions))
images_clean = list(set(images_paths))
images_clean  = [Image.open(path).convert("RGB") for path in images_clean]

# Validation loop
correct_predictions = 0
total_samples = 0
batch_size = 32

with torch.no_grad():
    for i in range(0, len(images_clean), batch_size):
        batch_images = images_clean[i:i + batch_size]
        batch_descriptions = descriptions[i:i + batch_size]
        indices_descriptions = [descriptions_clean.index(desc) for desc in batch_descriptions]

        inputs = processor(text=descriptions_clean, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Calculate loss
        ground_truth = torch.tensor(indices_descriptions, device=device)

        # Calculate accuracy
        preds = torch.argmax(logits_per_image, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(ground_truth.cpu().numpy())

# Creating a confusion matrix
num_classes = len(descriptions_clean)
confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

for t, p in zip(all_labels, all_preds):
    confusion_matrix[t, p] += 1

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(confusion_matrix, cmap='Blues')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(descriptions_clean, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticklabels(descriptions_clean)

# Loop over data dimensions and create text annotations.
for i in range(num_classes):
    for j in range(num_classes):
        text = ax.text(j, i, confusion_matrix[i, j].item(),
                       ha="center", va="center", color="black")

ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('./trains/siglip_confusion_matrix.png')
plt.show()