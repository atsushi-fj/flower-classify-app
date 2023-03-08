import os
import torch
from torch.utils.data import DataLoader


NUM_WORKERS = os.cpu_count()


def create_flowers_dataloaders(train_data,
                               val_data,
                               test_data,
                               batch_size=32,
                               seed=42,
                               num_workers=NUM_WORKERS):
    torch.manual_seed(seed)
    
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

    