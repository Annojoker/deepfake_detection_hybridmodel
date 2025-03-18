import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
from skimage import filters

# Paths
dataset_path = "/mnt/d/projects/seai/deepfake_detection_hybridmodel/dataset"
real_path = os.path.join(dataset_path, "real")
fake_path = os.path.join(dataset_path, "fake")
real_features_path = os.path.join(dataset_path, "real_features")
fake_features_path = os.path.join(dataset_path, "fake_features")

# Create output directories
os.makedirs(real_features_path, exist_ok=True)
os.makedirs(fake_features_path, exist_ok=True)

# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))  # Resize to match ViT/Xception input size
    
    # 1️⃣ **Fourier Transform**
    fourier = np.abs(fft2(img))  # Frequency representation
    
    # 2️⃣ **Local Binary Pattern (LBP)**
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")  # Extract LBP features
    
    # 3️⃣ **Edge Detection**
    edges = filters.sobel(img)  # Sobel edge detection
    
    # Stack all features into a single vector
    features = np.stack([fourier, lbp, edges], axis=0)  # Shape: (3, 224, 224)
    return features

# Process and save features
def process_dataset(input_dir, output_dir):
    for file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file)
        feature_array = extract_features(image_path)
        np.save(os.path.join(output_dir, file.replace(".jpg", ".npy")), feature_array)

# Run feature extraction
print("🔄 Extracting features from real images...")
process_dataset(real_path, real_features_path)

print("🔄 Extracting features from fake images...")
process_dataset(fake_path, fake_features_path)

print("✅ Feature extraction completed!")
