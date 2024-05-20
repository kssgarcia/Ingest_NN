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
            image = load_img(full_image_path, target_size=(256, 256))  # Adjust target size as needed
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

print(test_input.shape)
print(train_input.shape)
print(val_input.shape)
print(test_output.shape)
print(train_output.shape)
print(val_output.shape)

import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG-16 model without the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False

# Modify the output layer for 3000 classes
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
predictions = tf.keras.layers.Dense(3213, activation='sigmoid')(x)

# Create the final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(train_input, train_output, epochs=10, batch_size=32, validation_data=[val_input, val_output])