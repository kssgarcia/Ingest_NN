# %%

from roboflow import Roboflow
rf = Roboflow(api_key="ZguoagGlFgLAffjKef27")
project = rf.workspace("bri-institute-ns3xn").project("uascompvision")
version = project.version(1)
dataset = version.download("voc")


# %%
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load and parse dataset
def load_dataset(images_path, labels_path):
    images = []
    labels = []
    
    for image_file in os.listdir(images_path):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_path, image_file)
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_path, label_file)
            
            # Load image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            
            # Load label
            with open(label_path, 'r') as file:
                boxes = []
                for line in file.readlines():
                    _, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert from YOLO format to [xmin, ymin, xmax, ymax]
                    xmin = (x_center - width / 2) * 224
                    ymin = (y_center - height / 2) * 224
                    xmax = (x_center + width / 2) * 224
                    ymax = (y_center + height / 2) * 224
                    boxes.append([xmin, ymin, xmax, ymax])
                labels.append(boxes[0])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load the dataset
images_path = './uas_dataset/train/images'
labels_path = './uas_dataset/train/labels'
images, labels = load_dataset(images_path, labels_path)

# %%

# Normalize the images
images = images / 255.0

# Define the MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(4, activation='sigmoid')(x)  # 4 values for bounding box + 1 confidence score

# Define the final model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

# Define a callback to collect metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.evaluate(self.validation_data)

# Train the model
history = model.fit(images, labels, epochs=100, batch_size=1, validation_split=0.2)

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# %%

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model training and conversion to TFLite completed.")