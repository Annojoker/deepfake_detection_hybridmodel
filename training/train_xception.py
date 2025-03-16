import os
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Detect GPU or use CPU
physical_devices = tf.config.list_physical_devices('GPU')
USE_GPU = len(physical_devices) > 0

if USE_GPU:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ Using GPU for training")
    BATCH_SIZE = 32  # Standard batch size for GPU
else:
    print("⚠️ No GPU found, optimizing for CPU...")
    BATCH_SIZE = 16  # Reduce batch size for CPU efficiency
    tf.config.optimizer.set_jit(True)  # ✅ Enable XLA for CPU acceleration

# ✅ Define Training Parameters
IMG_SIZE = (224, 224)
EPOCHS = 10
DATASET_PATH = "../dataset"  # Ensure this contains 'real_processed' and 'fake_processed'

# ✅ Load pre-trained Xception model
base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained weights

# ✅ Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification (0=Real, 1=Fake)
])

# ✅ Compile model (outside device scope)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    classes=['real_processed', 'fake_processed'],  # ✅ Only select these two folders
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    classes=['real_processed', 'fake_processed'],  # ✅ Same for validation
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ✅ Confirm class mapping
print("Class mapping:", train_generator.class_indices)  # {'real_processed': 0, 'fake_processed': 1}

# ✅ Train Model
device = "/GPU:0" if USE_GPU else "/CPU:0"
with tf.device(device):
    print(f"🚀 Training on {device} (Batch Size: {BATCH_SIZE})...")
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# ✅ Save Model
model.save("../backend/models/xception_model.h5")
print("🎉 Training complete! Model saved at /backend/models/xception_model.h5")
