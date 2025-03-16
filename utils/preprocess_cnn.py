import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """Loads an image, resizes it, and normalizes it."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values
    return img

def preprocess_dataset(input_folder, output_folder):
    """Processes all images in a dataset folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = preprocess_image(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img * 255)  # Save processed image

# Example usage
preprocess_dataset("dataset/real", "dataset/real_processed")
preprocess_dataset("dataset/fake", "dataset/fake_processed")
    
    # In this code snippet, we define two functions:  preprocess_image  and  preprocess_dataset . The  preprocess_image  function loads an image, resizes it to the target size (224×224 pixels), and normalizes the pixel values to the range [0, 1]. The  preprocess_dataset  function processes all images in a dataset folder by calling the  preprocess_image  function on each image. 
    # We then use these functions to preprocess the images in the  real  and  fake  folders and save the processed images in the  real_processed  and  fake_processed  folders, respectively. 
    # Now that we have preprocessed the images, we can proceed to the next step: loading the preprocessed images and their corresponding labels. 
    # Step 2: Load Preprocessed Images and Labels 
    # In this step, we will load the preprocessed images and their corresponding labels from the  real_processed  and  fake_processed  folders. We will assign labels of 0 to real images and labels of 1 to fake images. 
    # Here is the code to load the preprocessed images and labels: