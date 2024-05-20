import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import shutil
import os
from pathlib import Path

# Setup model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda' if torch.cuda.is_available() else 'cpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define directories and set batch size
directory_path = "../datasets/DatasetFood176"
root_dir = Path(f"{directory_path}/train").resolve()
images_paths = list(root_dir.rglob('*.jpg')) + list(root_dir.rglob('*.png'))
images_paths = images_paths[:50]  # Limit to first 10 images for demonstration
batch_size = 2  # Set the batch size

labels = [name for name in os.listdir(f"{directory_path}/train") if os.path.isdir(os.path.join(f"{directory_path}/train", name))]
descriptions = [f"a photo of {label}, a type of food" for label in labels]

# Ensure target directory exists
target_directory = f"{directory_path}/wrong"
os.makedirs(target_directory, exist_ok=True)

n_worst = 0
for i in range(0, len(images_paths), batch_size):
    batch_paths = images_paths[i:i+batch_size]
    # Load batch images and prepare descriptions
    batch_images = []
    batch_texts = []
    for path in batch_paths:
        try:
            with Image.open(path) as img:
                img = img.convert('RGB') 
                batch_images.append(img)
                batch_texts.extend(descriptions)  
        except UnidentifiedImageError:
            print(f"Failed to open image {path}, skipping.")
            continue

    if not batch_images: 
        continue

    # Prepare inputs and move them to the device
    inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    predicted_indices = torch.argmax(probs, dim=1)

    # Move misclassified images
    for idx, path in enumerate(batch_paths):
        if idx >= len(predicted_indices):  # If fewer images loaded than paths
            continue
        true_class = path.parent.name
        predicted_label = labels[predicted_indices[idx]]
        if true_class != predicted_label:
            n_worst += 1
            dest_directory = os.path.join(target_directory, true_class)
            os.makedirs(dest_directory, exist_ok=True)
            shutil.move(str(path), os.path.join(dest_directory, path.name))
            print(f"Moved misclassified image: {path} to {dest_directory}")

    # Clear CUDA cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Processed batch {i//batch_size + 1}/{(len(images_paths) + batch_size - 1) // batch_size} on {model.device}. With {n_worst} worst images.")

print("Processing complete.")