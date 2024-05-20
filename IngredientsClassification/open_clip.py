# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import open_clip

# Define a custom dataset class
class RecipeDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        text = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, text

# Load your data
data = pd.read_csv("dataset/full_dataset.csv")
directory_path = "./Recipes10T"

# Assuming 'title' column has labels and 'image_path' has image file paths
data['image_path'] = data['title'].map(lambda title: os.path.join(directory_path, title))

# Split the data into training and validation sets
train_data, val_data = np.split(data.sample(frac=1, random_state=42), [int(0.8 * len(data))])

# Setup model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Create datasets and dataloaders
train_dataset = RecipeDataset(train_data, directory_path, transform=preprocess)
val_dataset = RecipeDataset(val_data, directory_path, transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.2)
loss_fn = open_clip.loss.ClipLoss()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, texts in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        texts = tokenizer(texts).to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = loss_fn(outputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, texts in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            texts = tokenizer(texts).to(device)

            outputs = model(images, texts)
            loss = loss_fn(outputs)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"clip_model_epoch_{epoch+1}.pth")

print("Training complete.")
