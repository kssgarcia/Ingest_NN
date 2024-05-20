# %%
import torch
from ultralytics import YOLO

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = YOLO("yolov8n-cls.pt").to(device)

    model.train(project='FoodClassyScrapping', data='./testRecipes', epochs=30, imgsz=640, device=device)

if __name__ == "__main__":
    main()

