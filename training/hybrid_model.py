import os
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from transformers import TFViTModel

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"
image_size = (224, 224)
batch_size = 16

# Load ViT model
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# ViT expects pixel values normalized to [-1, 1]
def vit_custom_preprocess(image):
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image

# Preprocessing function
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)

    vit_input = vit_custom_preprocess(image)
    xception_input = image / 255.0

    return {"vit_input": vit_input, "xception_input": xception_input}, label

# Dataset loader
def build_dataset(directory):
    class_names = sorted(os.listdir(directory))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    image_paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_index[class_name])

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Load datasets
train_ds = build_dataset(train_dir)
val_ds = build_dataset(val_dir)

# Inputs
vit_input = Input(shape=(224, 224, 3), name="vit_input")
xception_input = Input(shape=(224, 224, 3), name="xception_input")

# ViT branch
def extract_vit_features(images):
    vit_features = vit_model(images).last_hidden_state[:, 0]
    return vit_features

vit_output = Lambda(extract_vit_features, output_shape=(768,))(vit_input)
vit_output = BatchNormalization()(vit_output)

# Xception branch
xception_base = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
xception_output = xception_base(xception_input)
xception_output = GlobalAveragePooling2D()(xception_output)
xception_output = BatchNormalization()(xception_output)

# Combine and classify
combined = Concatenate()([vit_output, xception_output])
x = Dense(256, activation="relu")(combined)
x = BatchNormalization()(x)
output = Dense(1, activation="sigmoid")(x)

# Model build
model = Model(inputs=[vit_input, xception_input], outputs=output)

# 🔒 Optional: Freeze base for 3 epochs, then unfreeze
vit_model.trainable = False
xception_base.trainable = False

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Phase 1: Warm-up training
model.fit(train_ds, validation_data=val_ds, epochs=3)

# 🔓 Phase 2: Fine-tune full model
vit_model.trainable = True
xception_base.trainable = True

# Use a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=7)  # total 10 epochs (3 + 7)

# Save model
model.save("backend/models/hybrid_vit_xception_model.keras")
