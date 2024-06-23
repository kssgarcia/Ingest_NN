# %%
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

def get_images_paths(path): 
    return list(path.rglob('*.jpg')) + list(path.rglob('*.png'))

# Load your data
data = pd.read_csv("dataset/full_dataset.csv")
labels = data['title'].tolist()[:2000]

# %%
data[:2000].to_csv("./first_2000_rows.csv", index=False)

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTG3jTszSflQt-SjZGIWqJRegF0GrAVzpCQtg&s"
image = Image.open(requests.get(url, stream=True).raw)
plt.imshow(image)
plt.axis('off')
plt.show()

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
# %%
def tensor_memory(tensor):
    return tensor.element_size() * tensor.numel()

total_memory = sum(tensor_memory(v) for v in inputs.values())
print(f"Total memory usage: {total_memory / (1024 ** 2):.2f} MB")
# %%

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
top_probs, top_labels = torch.topk(probs, k=3, dim=1)

for prob, label in zip(top_probs[0], top_labels[0]):
    print(f"Prediction: {labels[label]} | Probability: {prob.item()}")
