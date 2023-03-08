import os
import torch
import torchvision 
from torchvision import datasets, transforms
from pathlib import Path

import data_setup
import engine
from utils import prepare_labels_list, EarlyStopping
from model_building import CCT


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
NUM_WORKERS = os.cpu_count()

# Setup directories
data_dir = Path("data")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] working on {device}")

# Set seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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
file_name = "class_names.txt"
class_names = prepare_labels_list(file_name)

# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, test_dataloader = data_setup.create_flowers_dataloaders(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    batch_size=BATCH_SIZE,
    seed=SEED,
    num_workers=NUM_WORKERS
)

# Create model with help from cct.py
model = CCT(
    embedding_dim=EMBEDDING_DIM,
    n_classes=N_CLASSES
)

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

