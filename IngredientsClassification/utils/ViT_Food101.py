# %%
from transformers import TrainingArguments, Trainer
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

# Define the paths to your dataset
train_data_path = "./food_101/train"
val_data_path = "./food_101/val"

# Load training and validation datasets
dataset_train = datasets.ImageFolder(root=train_data_path, transform=feature_extractor)
dataset_val = datasets.ImageFolder(root=val_data_path, transform=feature_extractor)

for parameter in model.parameters():
    parameter.requires_grad = False

class_names = dataset_train.classes
model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# Training parameters (adjust as needed)
learning_rate = 1e-5
batch_size = 4
num_epochs = 10

# Create DataLoaders for training and validation
train_dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    dataset_val, batch_size=batch_size, shuffle=False, drop_last=False
)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

summary(model=model, input_size=(4, 3, 224, 224))

# %%

# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image['pixel_values'][0]
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)['logits']
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.requires_grad = True
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image['pixel_values'][0]
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)['logits']
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
epochs = 1
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_dataloader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, val_dataloader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    
# save the trained model weights
model.save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
model.save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('TRAINING COMPLETE')