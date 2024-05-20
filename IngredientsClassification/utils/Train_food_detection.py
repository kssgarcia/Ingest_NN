# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Leer el archivo de texto y almacenar los ingredientes en una lista
with open('./datasets/Recipes5k/annotations/ingredients_Recipes5k.txt', 'r') as file:
    recipes = [linea.strip().split(',') for linea in file.readlines()]

ingredients_list = list(set([i for line in recipes for i in line]))
ingredients_vec = np.array([element for element in ingredients_list if element != ""])

output = np.zeros((ingredients_vec.shape[0],))

def compute_output(recipes, labelspath):
    labels = np.loadtxt(labelspath, dtype=int)
    output = np.zeros((labels.shape[0], ingredients_vec.shape[0]))
    for i, label in enumerate(labels):
        positions = np.where(np.isin(ingredients_vec, recipes[label]))[0] 
        output[i, positions] = 1
    return output

def compute_input(filepath):
    images_dir = './datasets/Recipes5k/images/'
    # Read the file and extract the image paths
    with open(filepath, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]

    images = []
    # Iterate over the image paths, load each image, and preprocess it
    for i,image_path in enumerate(image_paths):
        full_image_path = os.path.join(images_dir, image_path)

        if os.path.exists(full_image_path):
            image = load_img(full_image_path, target_size=(224, 224))  # Adjust target size as needed
            image = img_to_array(image)
            image = image / 255.0  # Normalize pixel values to [0, 1]

            images.append(image)

    images = np.array(images)

    return images

test_input = './datasets/Recipes5k/annotations/test_images.txt'  
val_input = './datasets/Recipes5k/annotations/val_images.txt'  
train_input = './datasets/Recipes5k/annotations/train_images.txt'  

test_output = './datasets/Recipes5k/annotations/test_labels.txt'  
val_output = './datasets/Recipes5k/annotations/val_labels.txt'  
train_output = './datasets/Recipes5k/annotations/train_labels.txt'  

test_input = compute_input(test_input)
test_output = compute_output(recipes, test_output)
val_input = compute_input(val_input)
val_output = compute_output(recipes, val_output)
train_input = compute_input(train_input)
train_output = compute_output(recipes, train_output)

# %%

from ultralytics import YOLO
from torchinfo import summary

model = YOLO("yolov8m-cls.pt")

if __name__ == "__main__":
    model.train(project='FoodtoFood', data='./datasets/food_101', epochs=30, imgsz=256)


# %%
# Define the number of sample images to visualize
num_samples = 5

# Create a figure and axis object to plot the images
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

# Iterate over a few sample images and labels in the dataset
for i in range(num_samples):

    # Plot the image
    axes[i].imshow(train_input[200])
    axes[i].set_title(train_output[200])
    axes[i].axis('off')

# Display the plot
plt.tight_layout()
plt.show()

# %%

from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow import keras

def custom_model(input_shape, output_shape):
    input_layer = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer

    # Add fully connected layers
    x = Dense(4096, activation='relu', name='fc_food')(x)
    food_out = Dense(output_shape[0], activation='softmax', name='food_out')(x)

    # Define the model with input and output layers
    model = Model(inputs=input_layer, outputs=food_out)

    # Compile the model
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model

# Example usage:
input_shape = (224, 224, 3)
output_shape = (3213,)  # Adjust according to your output shape

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    './bestResNet/cp.ckpt',
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose= 1,
)

# Create the custom model
model = custom_model(input_shape, output_shape)
history = model.fit(train_input, train_output, epochs=100, batch_size=10, validation_data=[val_input, val_output], callbacks=[checkpoint_callback])

model.save('./modelResNet')

dir = './plots'
if not os.path.exists(dir): os.makedirs(dir)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'][1:], label='Validation Loss')
plt.title('U-Net Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_plot_sparseResNet.png')  # Save the plot as an image
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('U-Net Training  Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_plot_sparseResNet.png')  # Save the plot as an image
plt.show()
