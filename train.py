"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import subprocess
import torch
import torchvision 
from torchvision import datasets, transforms
from pathlib import Path

import data_setup
import engine
from utils import prepare_labels_list, EarlyStopping
from model_building import cct


# setup hyperparameters
NUM_EPOCHS = 10000
IMG_SIZE = 224
EMBEDDING_DIM = 256
SEED = 42
BATCH_SIZE = 96
LEARNING_RATE = 5e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 6e-2
PATIENCE = 200
N_CLASSES = 102


# Setup directories
data_dir = Path("data")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create trasformers
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Download Flowers102 dataset
train_data = datasets.Flowers102(
    root=data_dir,
    split="train",
    transform=train_transforms,
    download=True,
)

val_data = datasets.Flowers102(
    root=data_dir,
    split="val",
    transform=test_transforms,
    download=True
)

test_data = datasets.Flowers102(
    root=data_dir,
    split="test",
    transform=test_transforms,
    download=True
)

# Get class name
url = "https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt"
cmd = f"wget {url}"
subprocess.call(cmd.split())
file_name = "Oxford-102_Flower_dataset_labels.txt"
class_names = prepare_labels_list(file_name)

# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_data,
    val_dir=val_data,
    test_dir=test_data,
    batch_size=BATCH_SIZE,
    seed=SEED
)

# Create model with help from cct.py
model = cct.CCT(
    img_size=IMG_SIZE,
    embedding_dim=EMBEDDING_DIM,
    n_classes=N_CLASSES
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                             lr=LEARNING_RATE,
                             betas=BETAS,
                             weight_decay=WEIGHT_DECAY)

# Set early stopping with help from 
earlystopping = EarlyStopping(patience=PATIENCE, verbose=True)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             earlystopping=earlystopping,
             device=device)
