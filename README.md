# FoodnotFood

- Updates February 13 2025
  - Implemented MobileNetV2 classification model
  - Added real-time camera processing
  - Optimized model size to 4MB

## Features

- Lightweight model (4MB)
- Real-time classification
- On-device processing
- Plate detection and cropping
- Optimized for mobile devices

## Project Structure

```
FoodnotFood/
├── models/
│   ├── classifier.tflite            # Main classification model
│   ├── food_detector_uint8.tflite   # Alternative model
│   └── android.tflite              # Android-optimized model
├── scripts/
│   ├── cropPlate.py                # Image preprocessing
│   ├── loadtflite.py               # Model testing utilities
│   ├── MobileNet.py                # Model architecture
│   └── organized_dataset.py        # Dataset organization
└── notebooks/
    └── Model_Maker_Object_Detection.ipynb
```

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Android Studio (for mobile deployment)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/FoodnotFood.git
cd FoodnotFood
```
