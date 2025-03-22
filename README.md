# Plant Disease Classification using ResNet50

## Description
This project implements a plant disease classification pipeline using transfer learning with ResNet50 in PyTorch. The system includes data augmentation, early stopping, and GPU acceleration to efficiently train and evaluate the model on a dataset of plant images. The model is designed to diagnose plant diseases by classifying input images into one of the predefined disease classes.

## Features
- **Transfer Learning**: Utilizes a pre-trained ResNet50 model fine-tuned on a plant disease dataset.
- **Data Augmentation**: Applies random resized cropping, horizontal flipping, and rotation to enhance the training data.
- **Early Stopping**: Implements early stopping to prevent overfitting and reduce training time.
- **GPU Acceleration**: Leverages GPU when available for faster training and inference.
- **Inference Functionality**: Provides functions to load the trained model and perform predictions on new images.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Albumentations
- OpenCV
- Matplotlib
- NumPy
- Pandas

## Installation and Setup

### Requirements
- Python 3.8 or later.
- Google Colab environment or local machine with access to a GPU (optional).

### Installing Dependencies
Install the required libraries using pip:
```sh
pip install -q kaggle torch torchvision torchaudio albumentations opencv-python matplotlib numpy pandas
```

## Dataset Extraction
1. Upload the dataset ZIP file (`new-plant-diseases-dataset.zip`) to your Google Drive.
2. The script mounts Google Drive and extracts the dataset to `/content/dataset` if not already extracted.
3. Adjust the paths in the script if necessary.

## Directory Structure
The dataset is expected to have the following structure:
```
/content/dataset/
└── New Plant Diseases Dataset(Augmented)
    ├── New Plant Diseases Dataset(Augmented)
        ├── train/
        └── valid/
```

## Training the Model
- The script defines training and validation transforms.
- Data is loaded using `torchvision.datasets.ImageFolder`.
- A ResNet50 model is fine-tuned on the training data.
- Early stopping is applied based on validation accuracy.
- The best model is saved as `best_model.pth`.

## Inference
- The function `load_trained_model()` loads the best saved model.
- The function `predict_image(image_path)` processes a given image and outputs the predicted disease class.

## Running the Code
To train the model, run the script in a Python environment (e.g., Google Colab). Make sure to update the file paths as needed.

