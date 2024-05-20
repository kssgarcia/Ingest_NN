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
from transformers import CLIPModel, CLIPProcessor
import optuna

def get_images_paths(path): 
    return list(path.rglob('*.jpg')) + list(path.rglob('*.png'))

# Load your data
data = pd.read_csv("dataset/full_dataset.csv")
directory_path = "./Recipes10T"

labels_to_images = {name: get_images_paths(Path(f"{directory_path}/test/"+name).resolve()) for name in os.listdir(f"{directory_path}/test") if os.path.isdir(os.path.join(f"{directory_path}/test", name))}
labels = [name for name in os.listdir(f"{directory_path}/test") if os.path.isdir(os.path.join(f"{directory_path}/test", name))]

matching_rows = data[data['title'].isin(labels)]
matching_rows['images_paths'] = matching_rows['title'].map(labels_to_images)

# Flatten the dataset for DataLoader
image_paths = []
texts = []
for _, row in matching_rows.iterrows():
    for img_path in row['images_paths']:
        image_paths.append(img_path)
        texts.append(row['title'])
        texts.append(f"A food call {row['title']}. Made with {row['NER']}. With a recipe {row['ingredients']}")

# Convert to numpy arrays for indexing
image_paths = np.array(image_paths)
texts = np.array(texts)

# Create a random permutation of indices
num_samples = len(image_paths)
indices = np.random.permutation(num_samples)

# Calculate the split index
split_idx = int(0.8 * num_samples)

# Split the data
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_image_paths = image_paths[train_indices].tolist()
train_texts = texts[train_indices].tolist()
val_image_paths = image_paths[val_indices].tolist()
val_texts = texts[val_indices].tolist()

# %%

# Load the processor and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the datasets
class RecipeDataset(Dataset):
    def __init__(self, image_paths, texts):
        self.image_paths = image_paths
        self.texts = texts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        return image, text

def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
    return inputs

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    beta1 = trial.suggest_uniform("beta1", 0.8, 0.99)
    beta2 = trial.suggest_uniform("beta2", 0.9, 0.999)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    train_dataset = RecipeDataset(train_image_paths, train_texts)
    val_dataset = RecipeDataset(val_image_paths, val_texts)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    num_epochs = 5  # Shorten for tuning
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # Calculate loss
            ground_truth = torch.arange(len(logits_per_image),dtype=torch.long,device=device)
            loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) +
                    torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
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

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss

# Create an Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
