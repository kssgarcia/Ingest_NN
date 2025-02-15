# %%
import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
model_path = 'model_logs/model_v1.1_small/food_detector_mobilenetv3_v1.1_small.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(1) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    #  MobilenetV4
    # input_shape = input_details[0]['shape']
    # input_data = cv2.resize(frame, (input_shape[2], input_shape[3])) 
    # input_data = np.transpose(input_data, (2, 0, 1)) 
    # input_data = np.expand_dims(input_data, axis=0)  
    # input_data = input_data.astype(np.float32) / 255.0

    #  MobilenetV3
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data * 255.0).astype(np.uint8)  # Convert to UINT8

    print("Fixed input shape:", input_data.shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    
    # Perform post-processing (if needed) and display results
    # Example: You can print the output data or draw bounding boxes on the frame
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()