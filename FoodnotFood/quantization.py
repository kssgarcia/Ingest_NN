# %%
import tensorflow as tf

# Load the existing TensorFlow model
graph_def_file = "best3.pb"
input_arrays = ["input"]
output_arrays = ["output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
