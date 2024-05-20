# %% Crop image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image

# tensor([[ 46.4638, 422.7108,  87.0763, 150.4807],
#         [429.0220, 378.8631, 582.6051, 305.1798],
#         [ 43.5799, 822.2684,  83.0341, 153.3796],
#         [ 45.8823, 224.2131,  86.7399, 157.7769],
#         [ 45.6860, 622.2863,  86.8161, 156.1160]], device='cuda:0')

boxes = [429.0220, 378.8631, 582.6051, 305.1798]
transform = transforms.Compose([
    transforms.Resize((640,640)), 
    transforms.ToTensor(),  
])

image = Image.open('./badpic.jpg')
img = transform(image)
center_x, center_y, width, height = boxes
left = center_x - width / 2
top = center_y - height / 2

cropped_image = crop(image, top=int(top), left=int(left), height=int(height), width=int(width))
plt.imshow(cropped_image)
plt.show()
# %%

from ultralytics import YOLO

model = YOLO("./scrap100.pt")

results = model(cropped_image)

names = results[0].names
probs = results[0].probs.top5
conf = results[0].probs.top5conf.tolist()
print([{'prediction': names[key], 'prob': conf[i]} for i,key in enumerate(probs)])


