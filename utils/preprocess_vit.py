from transformers import ViTFeatureExtractor
import cv2
import os
import numpy as np

# Load ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess_image_vit(image_path):
    """Loads and preprocesses an image for ViT."""
    img = cv2.imread(image_path)  # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    inputs = feature_extractor(images=img, return_tensors="np")  # Apply ViT preprocessing
    return inputs["pixel_values"][0]  # Extract normalized tensor

def preprocess_dataset_vit(input_folder, output_folder):
    """Processes all images in a dataset folder for ViT."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img_array = preprocess_image_vit(img_path)
        np.save(os.path.join(output_folder, filename.replace(".jpg", ".npy")), img_array)  # Save as .npy file

# Example usage
preprocess_dataset_vit("deepfake_detection_hybridmodel/dataset/real", "deepfake_detection_hybridmodel/dataset/real_vit")
preprocess_dataset_vit("deepfake_detection_hybridmodel/dataset/fake", "deepfake_detection_hybridmodel/dataset/fake_vit")

