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
directory_path = "./Recipes10T"

labels_to_images = {name: get_images_paths(Path(f"{directory_path}/train/"+name).resolve()) for name in os.listdir(f"{directory_path}/train") if os.path.isdir(os.path.join(f"{directory_path}/train", name))}

# Flatten the dataset for DataLoader
images_paths = []
descriptions = []

for title, paths in labels_to_images.items():
    matching = data[data['title'] == title]
    if matching.empty: continue
    for path in paths:
        desc = f"A food called {title}. Made with {matching['NER'].values[0]}. With a recipe {matching['ingredients'].values[0]}"
        images_paths.append(path)
        descriptions.append(desc.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("-", ""))

# Combine paths and texts into a DataFrame
dataset_df = pd.DataFrame({'images_path': images_paths, 'descriptions': descriptions})

# Split the data
num_samples = len(dataset_df)
split_idx = int(0.8 * num_samples)
indices = np.random.permutation(num_samples)
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_data = dataset_df.iloc[train_indices]
val_data = dataset_df.iloc[val_indices]

# Setup model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# from transformers import AutoProcessor, AutoModel
# model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
# processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Freeze all parameters except for the last few layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the last transformer block
for param in model.text_model.encoder.layers[-2:].parameters():
    param.requires_grad = True

for param in model.vision_model.encoder.layers[-2:].parameters():
    param.requires_grad = True

class RecipeDataset(Dataset):
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        text = self.data.iloc[idx, 1]
        
        return image, text

def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
    return inputs

# Load your data
# Create datasets
train_dataset = RecipeDataset(train_data, directory_path)
val_dataset = RecipeDataset(val_data, directory_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Ensure the model is in evaluation mode
model.eval()

# Initialize lists to collect predictions and ground truths
all_preds = []
all_labels = []

# Validation loop
val_loss = 0.0
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Running Validation"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Calculate loss
        ground_truth = torch.arange(len(logits_per_image), device=device)
        loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) + 
                torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2

        val_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(logits_per_image, dim=1)
        correct_predictions += (preds == ground_truth).sum().item()
        total_samples += len(logits_per_image)

        # Collect predictions and ground truths
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(ground_truth.cpu().numpy())

avg_val_loss = val_loss / len(val_loader)
accuracy = correct_predictions / total_samples
print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Creating a confusion matrix
num_classes = max(max(all_preds), max(all_labels)) + 1
confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

for t, p in zip(all_labels, all_preds):
    confusion_matrix[t, p] += 1

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(confusion_matrix, cmap='Blues')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
