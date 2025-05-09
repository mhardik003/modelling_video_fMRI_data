# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import logging
from tqdm import tqdm

import config
from config import DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS

logger = logging.getLogger(__name__)

# def prepare_dataloaders(fmri_data: np.ndarray, video_embeddings: np.ndarray, batch_size: int = BATCH_SIZE):
#     """Creates train, validation, and test DataLoaders."""
#     logger.info(f"Preparing DataLoaders. Input fMRI: {fmri_data.shape}, Video: {video_embeddings.shape}")
#     if fmri_data.shape[0] != video_embeddings.shape[0]:
#         raise ValueError(f"Mismatch in time points between fMRI ({fmri_data.shape[0]}) and Video ({video_embeddings.shape[0]})")

#     dataset = TensorDataset(torch.from_numpy(fmri_data).float(),
#                             torch.from_numpy(video_embeddings).float())

#     total_size = len(dataset)
#     train_size = int(TRAIN_RATIO * total_size)
#     val_size = int(VAL_RATIO * total_size)
#     test_size = total_size - train_size - val_size

#     # Ensure split doesn't lose data due to rounding
#     if train_size + val_size + test_size < total_size:
#         train_size += total_size - (train_size + val_size + test_size)

#     logger.info(f"Splitting data: Train={train_size}, Val={val_size}, Test={test_size}")
#     # IMPORTANT: For time series, random_split is NOT ideal as it breaks temporal dependencies.
#     # A better approach is sequential splitting.
#     # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#     # Sequential split:
#     indices = np.arange(total_size)
#     train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
#     val_dataset = torch.utils.data.Subset(dataset, indices[train_size:train_size + val_size])
#     test_dataset = torch.utils.data.Subset(dataset, indices[train_size + val_size:])

#     logger.info(f"Train dataset size: {len(train_dataset)}")
#     logger.info(f"Validation dataset size: {len(val_dataset)}")
#     logger.info(f"Test dataset size: {len(test_dataset)}")

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, device, task='encoding'):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Training ({task})", leave=False)
    for fmri_batch, video_batch in progress_bar:
        fmri_batch = fmri_batch.to(device)
        video_batch = video_batch.to(device)

        optimizer.zero_grad()

        if task == 'encoding':
            # Predict fMRI from Video
            predictions = model(video_batch)
            loss = criterion(predictions, fmri_batch)
        elif task == 'decoding':
            # Predict Video from fMRI
            predictions = model(fmri_batch)
            loss = criterion(predictions, video_batch)
        else:
            raise ValueError("Task must be 'encoding' or 'decoding'")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, criterion, device, task='encoding'):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation ({task})", leave=False)
        for fmri_batch, video_batch in progress_bar:
            fmri_batch = fmri_batch.to(device)
            video_batch = video_batch.to(device)

            if task == 'encoding':
                predictions = model(video_batch)
                loss = criterion(predictions, fmri_batch)
            elif task == 'decoding':
                predictions = model(fmri_batch)
                loss = criterion(predictions, video_batch)
            else:
                raise ValueError("Task must be 'encoding' or 'decoding'")

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


    avg_loss = total_loss / num_batches
    return avg_loss


# def run_training(model, train_loader, val_loader, device, task='encoding', epochs=EPOCHS, learning_rate=LEARNING_RATE):
#     """Main training loop with validation."""
#     logger.info(f"Starting {task} model training for {epochs} epochs on {device}")
#     model.to(device)
#     criterion = nn.MSELoss() # Mean Squared Error is common for regression
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     best_val_loss = float('inf')
#     best_model_state = None

#     for epoch in range(epochs):
#         train_loss = train_epoch(model, train_loader, criterion, optimizer, device, task)
#         val_loss = validate_epoch(model, val_loader, criterion, device, task)

#         logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict()
#             logger.info(f"*** New best validation loss: {best_val_loss:.4f}. Saving model state. ***")

#     # Load best model state
#     if best_model_state:
#         model.load_state_dict(best_model_state)
#         logger.info("Loaded best model state based on validation loss.")

#     return model # Return the best model

def run_training_no_val(model, train_loader, device, task='encoding', epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    """ Simplified training loop without validation. """
    logger.info(f"Starting {task} model training (MLP, no validation) for {epochs} epochs on {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, task) # Assumes train_epoch is reverted
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        # No validation step, no saving best model based on validation

    logger.info("Training complete. Using model from last epoch.")
    return model # Return model from last epoch
