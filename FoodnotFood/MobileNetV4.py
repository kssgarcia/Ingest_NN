# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.onnx as onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# %%

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

data_dir = '../../Food-5K'
image_datasets = {
    'train': datasets.ImageFolder(root=f'{data_dir}/organized_train', transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=f'{data_dir}/organized_val', transform=data_transforms['val'])
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

# %%

model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%

num_epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

print('Training complete')

# %%

torch.save(model.state_dict(), 'mobilenetv4_binary.pth')
# %%

# Load the model
model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load('mobilenetv4_binary.pth'))

# %%
# Assuming 'model' is your trained MobileNetV4 model
model.eval()

# Create a dummy input tensor with the same size as your input data
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
onnx_model_path = 'mobilenetv4.onnx'
onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'], opset_version=12)

# %%

# Load the ONNX model
onnx_model = onnx.load('mobilenetv4.onnx')

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_model_path = 'mobilenetv4_tf'
tf_rep.export_graph(tf_model_path)

# %%
import tensorflow as tf

# Load the saved TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model('mobilenetv4_tf')

# Enable optimizations (optional but recommended for mobile)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'mobilenetv4.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved as {tflite_model_path}")