import torch
import numpy as np
import os
import tensorflow as tf
from transformers import ViTForImageClassification, ViTFeatureExtractor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Paths to trained models & datasets
MODEL_DIR = "./backend/models"
DATASET_DIR = "./dataset"
OUTPUT_DIR = "./dataset/extracted_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Load trained Xception model
xception_model = load_model(os.path.join(MODEL_DIR, "xception_model.h5"))

# ✅ Load trained ViT model
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
vit_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vit_model.pth")))
vit_model.eval()

# ✅ ViT Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_xception_features(img_dir, output_file):
    features = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))  # Xception input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        
        feat = xception_model.predict(img_array)[0]
        features.append(feat)
    
    np.save(output_file, np.array(features))
    print(f"✅ Saved Xception features: {output_file}")

def extract_vit_features(img_dir, output_file):
    features = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))  # ViT input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = feature_extractor(images=img_array, return_tensors="pt")["pixel_values"]
        
        with torch.no_grad():
            feat = vit_model(img_array).logits.numpy()[0]
            features.append(feat)
    
    np.save(output_file, np.array(features))
    print(f"✅ Saved ViT features: {output_file}")

# ✅ Extract and Save Features
extract_xception_features(os.path.join(DATASET_DIR, "real_processed"), os.path.join(OUTPUT_DIR, "real_xception.npy"))
extract_xception_features(os.path.join(DATASET_DIR, "fake_processed"), os.path.join(OUTPUT_DIR, "fake_xception.npy"))

def merge_vit_features(src_dir, output_file):
    features = []
    for npy_file in os.listdir(src_dir):
        npy_path = os.path.join(src_dir, npy_file)
        feat = np.load(npy_path)  # Load the extracted ViT feature
        features.append(feat)

    np.save(output_file, np.array(features))
    print(f"✅ Merged ViT features: {output_file}")

merge_vit_features(os.path.join(DATASET_DIR, "real_vit"), os.path.join(OUTPUT_DIR, "real_vit.npy"))
merge_vit_features(os.path.join(DATASET_DIR, "fake_vit"), os.path.join(OUTPUT_DIR, "fake_vit.npy"))


print("🎯 Feature extraction complete!")
