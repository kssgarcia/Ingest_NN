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
from torchvision import transforms

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
        # desc = f"A food called {title}."
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

from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)


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
        transform = transforms.Resize((224, 224))
        image = transform(Image.open(img_name).convert("RGB"))
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

# Set up optimizer
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of parameters: ", total_params)
print("Number of trainable parameters: ", trainable_params)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Initialize lists to track metrics
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate loss
        ground_truth = torch.arange(len(logits), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, ground_truth)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Calculate loss
            ground_truth = torch.arange(len(logits), device=device)
            loss = torch.nn.functional.cross_entropy(logits, ground_truth)

            val_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == ground_truth).sum().item()
            total_samples += len(logits)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save model and optimizer state after each epoch
model_save_path = f"./trains/vilt_model.pth"
optimizer_save_path = f"./trains/vilt_optimizer.pth"
torch.save(model.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)
print(f"Model and optimizer state saved after epoch {epoch+1}")

print("Training complete.")

# Plotting the metrics
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./trains/training_final_parameters.png')
plt.show()