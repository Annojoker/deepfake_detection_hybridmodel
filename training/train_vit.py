import torch
import os
import numpy as np
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# ✅ Check GPU Availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Load Pretrained ViT Model
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)  # 2 classes: Real/Fake
model.to(device)

# ✅ Configurations
BATCH_SIZE = 8    # Adjust based on GPU memory
NUM_EPOCHS = 5    # Total epochs

# ✅ Dataset Class (Loads data in mini-batches)
class ViTDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.data = []
        self.labels = []

        # ✅ Load all real and fake files
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
        
        self.file_paths = real_files + fake_files
        self.labels = [0] * len(real_files) + [1] * len(fake_files)  # 0 = Real, 1 = Fake

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = np.load(file_path)  # Load .npy image
        return {"pixel_values": torch.tensor(image, dtype=torch.float32).to(device),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long).to(device)}

# ✅ Define Dataset Paths
dataset_path = "/mnt/d/projects/seai/deepfake_detection_hybridmodel/dataset"
real_vit_dir = os.path.join(dataset_path, "real_vit")
fake_vit_dir = os.path.join(dataset_path, "fake_vit")

# ✅ Load Full Dataset
dataset = ViTDataset(real_vit_dir, fake_vit_dir)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./vit_model",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_steps=1000,
    logging_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        "pixel_values": torch.stack([item["pixel_values"].to("cpu") for item in data]),  # Move to CPU before pinning
        "labels": torch.tensor([item["labels"] for item in data], dtype=torch.long).to("cpu"),  # Ensure CPU tensor
    }
)

# ✅ Train Model
print("🚀 Starting ViT Training...")
trainer.train()

# ✅ Save Model
model_path = "/mnt/d/projects/seai/deepfake_detection_hybridmodel/backend/models/vit_model.pth"
torch.save(model.state_dict(), model_path)
print(f"🎉 Training complete! Model saved at {model_path}")
