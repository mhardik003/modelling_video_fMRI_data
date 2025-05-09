# src/evaluate.py
import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import logging
from tqdm import tqdm

from config import DEVICE, EVAL_METRICS

logger = logging.getLogger(__name__)

def calculate_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> (float, float):
    """Calculates Pearson correlation averaged across features/voxels."""
    # y_true, y_pred: shape (n_samples, n_features)
    correlations = []
    p_values = []
    if y_true.ndim == 1: # Single feature/voxel
        r, p = pearsonr(y_true, y_pred)
        return r if np.isfinite(r) else 0.0 , p
    
    for i in range(y_true.shape[1]):
        # Ignore features/voxels with no variance
        if np.var(y_true[:, i]) > 1e-6 and np.var(y_pred[:, i]) > 1e-6:
            r, p = pearsonr(y_true[:, i], y_pred[:, i])
            if np.isfinite(r): # Check for NaN correlation (can happen with constant data)
                 correlations.append(r)
                 p_values.append(p)
            else:
                 correlations.append(0.0) # Or handle as NaN/ignore? Assign 0 for now.
                 p_values.append(1.0)

        else:
             correlations.append(0.0) # No correlation if no variance
             p_values.append(1.0)


    avg_r = np.mean(correlations) if correlations else 0.0
    # For p-value, maybe report median or range? Average is less meaningful.
    # Let's return average r and list of all r's for potential mapping
    return avg_r, np.array(correlations)


def evaluate_model(model, dataloader, criterion, device, task='encoding'):
    """Evaluates the model on a given dataloader."""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        print(f"Evaluating {task} model...")
        progress_bar = tqdm(dataloader, desc=f"Evaluation ({task})", leave=False)
        for fmri_batch, video_batch in progress_bar:
            fmri_batch = fmri_batch.to(device)
            video_batch = video_batch.to(device)

            if task == 'encoding':
                predictions = model(video_batch)
                targets = fmri_batch
            elif task == 'decoding':
                predictions = model(fmri_batch)
                targets = video_batch
            else:
                raise ValueError("Task must be 'encoding' or 'decoding'")

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate batches
    predictions_np = np.concatenate(all_predictions, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    print(f"Evaluation complete. Target shape: {targets_np.shape}, Prediction shape: {predictions_np.shape}")

    # Calculate metrics
    results = {}
    if 'mse' in EVAL_METRICS:
        mse = mean_squared_error(targets_np, predictions_np)
        results['mse'] = mse
        print(f"  MSE: {mse:.4f}")

    if 'pearsonr' in EVAL_METRICS:
        avg_r, all_r = calculate_pearsonr(targets_np, predictions_np)
        results['pearsonr_avg'] = avg_r
        results['pearsonr_all'] = all_r
        print(f"  Avg Pearson Correlation (across features/voxels): {avg_r:.4f}")
        print(f"  Median Pearson Correlation: {np.median(all_r):.4f}")


    if 'r2' in EVAL_METRICS: # Optional: R-squared
         r2_uniform = r2_score(targets_np, predictions_np, multioutput='uniform_average')
         r2_variance = r2_score(targets_np, predictions_np, multioutput='variance_weighted')
         results['r2_uniform'] = r2_uniform
         results['r2_variance'] = r2_variance
         print(f"  R2 Score (uniform avg): {r2_uniform:.4f}")
         print(f"  R2 Score (variance weighted): {r2_variance:.4f}")

    return results, targets_np, predictions_np


def plot_predictions(targets, predictions, n_samples=5, feature_indices=None, title=""):
    """Plots target vs prediction for a few samples/features."""
    if feature_indices is None:
        # Select some random features/voxels to plot
        num_features = targets.shape[1]
        feature_indices = np.random.choice(num_features, min(n_samples, num_features), replace=False)

    n_plots = len(feature_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes] # Make iterable if only one plot

    time_points = np.arange(targets.shape[0])

    for i, feat_idx in enumerate(feature_indices):
        axes[i].plot(time_points, targets[:, feat_idx], label='Target', alpha=0.8)
        axes[i].plot(time_points, predictions[:, feat_idx], label='Prediction', alpha=0.8, linestyle='--')
        axes[i].set_ylabel(f'Feature {feat_idx}')
        axes[i].legend()
        axes[i].grid(True, linestyle=':')

    axes[-1].set_xlabel('Time Points')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig