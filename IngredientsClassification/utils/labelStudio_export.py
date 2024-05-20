# %%
from ultralytics import YOLO
from pathlib import Path
from torchvision import transforms
from PIL import Image
import pandas as pd

model = YOLO("../models/scrap20.pt")

root_dir = Path("./datasetScrapping/train").resolve()

# Recursively find all image files in the directory
image_paths = list(root_dir.rglob('*.jpg')) + list(root_dir.rglob('*.png'))

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((640,640)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])
# %%

data = []

for path in image_paths[:5]:
    image = Image.open(path)
    image = transform(image).unsqueeze(0)  
    result = model(image)
    true_class = path.parent.name
    predict_class = result[0].names[0]
    conf = result[0].probs.top5conf[0].tolist()

    # Append the data to the list
    data.append([str(path), true_class, predict_class, conf])

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['image_path', 'true_class', 'predict_class', 'conf'])
df = df.sort_values('conf')
lowest_conf_df = df.head()
# %%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Determine the number of rows and columns for the grid
num_images = len(lowest_conf_df)
num_cols = 3  # You can change this to fit your needs
num_rows = num_images // num_cols if num_images % num_cols == 0 else num_images // num_cols + 1

# Create a figure and axes for the grid
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Iterate over the DataFrame and display the images
for i, (index, row) in enumerate(lowest_conf_df.iterrows()):
    img = mpimg.imread(row['image_path'])
    ax = axs[i // num_cols, i % num_cols]
    ax.imshow(img)
    ax.set_title(f"TC: {row['true_class']}, PC: {row['predict_class']}, C: {row['conf']}")

# Remove empty subplots
if num_images % num_cols != 0:
    for ax in axs.flatten()[num_images:]:
        fig.delaxes(ax)

plt.tight_layout()
plt.show()