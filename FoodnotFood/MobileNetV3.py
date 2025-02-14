# %%
import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --------------------- #
#    Setup & Config     #
# --------------------- #

# Define model version (you can automate this or retrieve from a config)
model_version = "v1.0_small"

# Define a base logs directory (could also be made dynamic with timestamp)
log_base_dir = "model_logs"
log_dir = os.path.join(log_base_dir, f"model_{model_version}")
os.makedirs(log_dir, exist_ok=True)

# Define your dataset paths
train_dir = '../../Food-5K/organized_train'
val_dir = '../../Food-5K/organized_val'

# --------------------- #
#    Data Generators    #
# --------------------- #

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# %%

# --------------------- #
#   Model Definition    #
# --------------------- #

base_model = MobileNetV3Small(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Add a custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# %%

# --------------------- #
#       Training        #
# --------------------- #

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the standard Keras (SavedModel) in the logs directory
saved_model_dir = os.path.join(log_dir, "saved_model")
model.export(saved_model_dir)

# %%

# --------------------- #
#   Plot & Save Curves  #
# --------------------- #

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save the plot image into the log folder
plot_path = os.path.join(log_dir, "training_curves.png")
plt.savefig(plot_path)
plt.close()

# --------------------- #
#   Save Metrics JSON   #
# --------------------- #

# Here, we store final metrics. You can store the entire history if desired.
training_metrics = {
    "model_version": model_version,
    "final_training_accuracy": float(acc[-1]),
    "final_validation_accuracy": float(val_acc[-1]),
    "final_training_loss": float(loss[-1]),
    "final_validation_loss": float(val_loss[-1]),
    "epochs": len(epochs),
    "timestamp": datetime.datetime.now().isoformat()
}

metrics_path = os.path.join(log_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(training_metrics, f, indent=4)

# %%
# --------------------- #
#   TFLite Conversion   #
# --------------------- #

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(os.path.join(train_dir, '*', '*'))

    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    for image_path in dataset_list.take(100):
        yield [preprocess_image(image_path)]

# Create converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Save the TFLite model in the same log directory
tflite_model_path = os.path.join(log_dir, f"food_detector_mobilenetv3_{model_version}.tflite")
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"\nModel version '{model_version}' training complete.")
print(f"Logs and artifacts saved in: {log_dir}")
