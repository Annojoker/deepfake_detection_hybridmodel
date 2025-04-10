import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from keras.saving import register_keras_serializable
from transformers import TFAutoModel  # 👈 Missing import added

# App init
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ViT model
vit_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

@register_keras_serializable()
def extract_vit_features(images):
    images = tf.transpose(images, perm=[0, 3, 1, 2])  # Convert to NCHW format if needed
    vit_features = vit_model(images).last_hidden_state[:, 0]
    return vit_features

# Load the hybrid model
model = load_model("models/hybrid_vit_xception_model.keras")

# Custom preprocessing for ViT
def vit_preprocess(image):
    image = image / 255.0
    return (image - 0.5) / 0.5

# Image preprocessing
def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32")

    vit_input = vit_preprocess(img_array)
    xception_input = img_array / 255.0

    return {
        "vit_input": np.expand_dims(vit_input, axis=0),
        "xception_input": np.expand_dims(xception_input, axis=0)
    }

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    inputs = prepare_image(contents)
    prediction = model.predict(inputs)[0][0]
    label = "Fake" if prediction >= 0.5 else "Real"
    return {
        "filename": file.filename,
        "prediction": float(prediction),
        "label": label
    }
@app.get("/")
def read_root():
    return {"message": "Deepfake Detection API is up and running 🚀"}

# Run with: uvicorn backend.api:app --reload
if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
