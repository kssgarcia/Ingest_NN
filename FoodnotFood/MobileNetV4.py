# %%
import os
import json
import datetime
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.onnx as onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import matplotlib.pyplot as plt

# %% ------------------ #
#    Setup & Config     #
# --------------------- #

# Define model version
model_version = "v1.0_test"

# Define base logs directory
log_base_dir = "model_log_torch"
log_dir = os.path.join(log_base_dir, f"model_{model_version}")
os.makedirs(log_dir, exist_ok=True)

# Define dataset paths
data_dir = '../../Food-5K'

# Save model metadata
metadata = {
    "model_version": model_version,
    "timestamp": str(datetime.datetime.now()),
    "data_dir": data_dir,
    "log_dir": log_dir,
}
with open(os.path.join(log_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# %% ------------------ #
#    Data Preparation   #
# --------------------- #

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(root=f'{data_dir}/organized_train', transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=f'{data_dir}/organized_val', transform=data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

# %% ------------------ #
#    Model Setup        #
# --------------------- #

model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% ------------------ #
#    Training Loop      #
# --------------------- #

num_epochs = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

metrics = {"train": [], "val": []}

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
ud = []
activation_stats = {}

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        metrics[phase].append({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_acc.item()})

# Save training metrics
with open(os.path.join(log_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
torch.save(model.state_dict(), os.path.join(log_dir,'mobilenetv4_binary.pth'))

print('Training complete')

# ------------------ #
#   Plot & Save Curves  #
# ------------------ #

def plot_metrics(metrics, save_path):
    epochs = list(range(len(metrics['train'])))
    train_acc = [m['accuracy'] for m in metrics['train']]
    val_acc = [m['accuracy'] for m in metrics['val']]
    train_loss = [m['loss'] for m in metrics['train']]
    val_loss = [m['loss'] for m in metrics['val']]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.savefig(save_path)
    plt.show()

plot_metrics(metrics, os.path.join(log_dir, "training_curves.png"))
# %% ------------------ # 
#    Import Model     #
# --------------------- #
# Load the model
model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(os.path.join(log_dir, 'mobilenetv4_binary.pth')))


# %% ------------------ #
#    Export to ONNX     #
# --------------------- #
import torch
import torch.nn as nn
import torch.onnx as onnx

# Assuming 'model' is your trained MobileNetV4 model
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
onnx_model_path = os.path.join(log_dir, 'mobilenetv4.onnx')
onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'], opset_version=12)

# %% ------------------ #
#  Convert to TF Model  #
# --------------------- #
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_model_path = os.path.join(log_dir, 'mobilenetv4_tf')
tf_rep.export_graph(tf_model_path)

# %% ------------------ #
#  Convert to TFLite   #
# --------------------- #

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = os.path.join(log_dir, 'mobilenetv4.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved in {tflite_model_path}")
