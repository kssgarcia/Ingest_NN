# %%
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import pandas as pd

model = YOLO("../models/scrap174.pt")
#model = YOLO("../models/scrap100.pt")

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((640,640)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

image = Image.open('./img/rise.jpg')

result = model(transform(image))

names = result[0].names
probs = result[0].probs.top5
conf = result[0].probs.top5conf.tolist()
df = pd.DataFrame([{'prediction': names[key], 'prob': conf[i]} for i,key in enumerate(probs)])

# %%
model = YOLO("./best1.pt")