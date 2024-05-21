# %%
from PIL import Image
import os
import torch
import requests
import pandas as pd

# Setup model and processor
# from transformers import CLIPProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from transformers import AutoProcessor, AutoModel
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Load the model state dictionary
model_path = "./trains/siglip_model.pth"  # Replace with the correct path to your saved model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model to evaluation mode
model.eval()

# Load and preprocess an image
url = "https://assets.epicurious.com/photos/568eb0bf7dc604b44b5355ee/16:9/w_2560%2Cc_limit/rice.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = image.convert('RGB')

import matplotlib.pyplot as plt

# Plot the image
plt.imshow(image)
plt.axis('off')
plt.show()

# Load your data
data = pd.read_csv("dataset/full_dataset.csv")
directory_path = "./Recipes10T"

labels = [name for name in os.listdir(f"{directory_path}/train") if os.path.isdir(os.path.join(f"{directory_path}/train", name))]

# Flatten the dataset for DataLoader
descriptions = []

for title in labels:
    matching = data[data['title'] == title]
    if matching.empty: continue
    desc = f"A food called {title}. Made with {matching['NER'].values[0]}. With a recipe {matching['ingredients'].values[0]}"
    descriptions.append(desc.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("-", ""))

# Process the image and text
inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True, truncation=True).to(model.device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text

# Get the predicted labels and probabilities
probs = logits_per_image.softmax(dim=1)
prob_values, predicted_labels = torch.topk(probs, k=5)

# Print the last 5 predictions with probability values
for i in range(-1, -6, -1):
    prob = prob_values[0][i].item()
    label = predicted_labels[0][i].item()
    predicted_label = descriptions[label]
    print(f"Prediction: {predicted_label}")
    print(f"Probability: {prob}")
    print()

print(f"The model predicts: {predicted_label}")