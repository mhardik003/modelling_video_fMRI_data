# src/preprocessing.py
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import nilearn.glm.first_level as first_level # For HRF
from sklearn.feature_selection import SelectKBest, f_classif
from nilearn.glm.first_level import glover_hrf
import logging
from tqdm import tqdm

from config import ALIGN_METHOD, HRF_DELAY, TR, VIDEO_CHUNK_SIZE, VIDEO_CHUNK_STRIDE

logger = logging.getLogger(__name__)

def preprocess_fmri(DO_FMRI_ZSCORE, APPLY_PCA, PCA_N_COMPONENTS, fmri_data: np.ndarray) -> np.ndarray:
    """Applies z-scoring per voxel across time."""
    if DO_FMRI_ZSCORE:
        print("Applying Z-score normalization to fMRI data (per voxel).")
        fmri_data = zscore(fmri_data, axis=0, ddof=1)
        # Handle potential NaNs if voxel has zero variance
        fmri_data = np.nan_to_num(fmri_data)

    if APPLY_PCA:
        # apply PCA to reduce dimensionality to 10000 components to the data of shape (x,y) where I want to reduce the dimensionality of Y
        print(f"Applying PCA to fMRI data, reducing to {PCA_N_COMPONENTS} components.")

        voxel_variances = np.var(fmri_data, axis=0)
        
        top_variance_indices = np.argsort(voxel_variances)[-PCA_N_COMPONENTS:]
        fmri_reduced = fmri_data[:, top_variance_indices]

    

    return fmri_reduced

def preprocess_video_embeddings(DO_VIDEO_ZSCORE, video_embeddings: np.ndarray) -> np.ndarray:
    """Applies z-scoring per feature dimension across time."""
    if DO_VIDEO_ZSCORE:
        print("Applying Z-score normalization to video embeddings (per feature).")
        video_embeddings = zscore(video_embeddings, axis=0)
        # Handle potx1ential NaNs if feature has zero variance
        video_embeddings = np.nan_to_num(video_embeddings)
    return video_embeddings

def create_hrf_kernel(tr: float, oversampling: int = 16) -> np.ndarray:
    """Creates a canonical HRF kernel."""
    # Use nilearn's spm HRF + derivatives, or just the main one
    frame_times = np.arange(0, 32, tr / oversampling) # Create HRF over ~32 seconds at high resolution
    hrf = first_level.spm_hrf(tr / oversampling, frame_times=frame_times)[0] # Get main HRF
    # Normalize? Optional, convolution will scale output anyway.
    # hrf /= hrf.sum()
    return hrf

def align_data(fmri_data: np.ndarray,
               video_embeddings: np.ndarray,
               fmri_tr: float,
               video_fps: float,
               num_video_frames_original: int,
               hrf_delay: float = HRF_DELAY,
               align_method: str = ALIGN_METHOD,
               video_chunk_size: int = VIDEO_CHUNK_SIZE,
               video_chunk_stride: int = VIDEO_CHUNK_STRIDE
               ) -> (np.ndarray, np.ndarray):
    """
    Aligns fMRI and video data temporally.
    1. Calculates timestamps for both modalities.
    2. Handles HRF (shift or convolution).
    3. Resamples video embeddings to fMRI TR.
    """
    print(f"Starting temporal alignment: fMRI {fmri_data.shape}, Video Embeddings {video_embeddings.shape}")
    print(f"fMRI TR: {fmri_tr:.3f}s, Video FPS: {video_fps:.2f}, Alignment: {align_method}")

    # --- Calculate Timestamps ---
    # fMRI timestamps (center of each TR, relative to start of *kept* data)
    fmri_times = np.arange(fmri_data.shape[0]) * fmri_tr + fmri_tr / 2.0

    # Video embedding timestamps
    # Embeddings correspond to chunks. Calculate center time of each chunk.
    # Assume chunks are processed with 'video_chunk_stride'
    num_chunks = video_embeddings.shape[0]
    chunk_centers_frame_idx = np.arange(num_chunks) * video_chunk_stride + video_chunk_size / 2.0
    video_embedding_times = chunk_centers_frame_idx / video_fps

    print(f"fMRI time range: {fmri_times[0]:.2f}s to {fmri_times[-1]:.2f}s ({len(fmri_times)} points)")
    print(f"Video embedding time range: {video_embedding_times[0]:.2f}s to {video_embedding_times[-1]:.2f}s ({len(video_embedding_times)} points)")


    # --- HRF Adjustment (applied to video times) ---
    if align_method == 'shift':
        print(f"Applying HRF shift of {hrf_delay:.2f}s to video timestamps.")
        # video_embedding_times_aligned = video_embedding_times + hrf_delay
        video_embedding_times_aligned = np.roll(video_embedding_times, shift=hrf_delay, axis=0) # this will shift the array by hrf_delay


    elif align_method == 'convolve':
        # Convolve video embeddings with HRF
        print("> Applying HRF convolution")
        hrf = glover_hrf(tr=fmri_tr, oversampling=16) # Use Glover HRF # here the oversampling is set to 16 
        convolved = np.zeros_like(video_embeddings)

        for i in range(video_embeddings.shape[1]):  # Iterate over each feature
            convolved[:, i] = fftconvolve(video_embeddings[:, i], hrf, mode='full')[:video_embeddings.shape[0]]

        video_embedding_times_aligned = convolved

    else:
        logger.warning(f"Unknown align_method: {align_method}. No HRF adjustment applied.")
        video_embedding_times_aligned = video_embedding_times

    # match the first dimension of the video embeddings to the fMRI data
    if video_embedding_times_aligned.shape[0] != fmri_data.shape[0]:
        # Resample video embeddings to match fMRI TR
        if video_embedding_times_aligned.shape[0] > fmri_data.shape[0]:
            # Interpolate to resample video embeddings to fMRI TR
            print("Resampling video embeddings to match fMRI TR.")
            # remove the extra frames from the video embeddings
            video_embedding_times_aligned = video_embedding_times_aligned[:fmri_data.shape[0]]
            # video_embeddings_resampled = interp1d(video_embedding_times_aligned, video_embeddings, axis=0, fill_value="extrapolate")(fmri_times)
            video_embeddings_resampled = video_embedding_times_aligned
            fmri_data_aligned = fmri_data
        
        elif fmri_data.shape[0] > video_embedding_times_aligned.shape[0]:
            
            # remove the extra frames from the fMRI data
            fmri_data_aligned = fmri_data[:video_embedding_times_aligned.shape[0]]
            # Interpolate to resample fMRI data to video TR
            print("Resampling fMRI data to match video TR.")
            # fmri_data_aligned = interp1d(fmri_times, fmri_data_aligned, axis=0, fill_value="extrapolate")(video_embedding_times_aligned)
            video_embeddings_resampled = video_embeddings

    else:
        # If they are already aligned, just use the original data

        print("No resampling needed, data is already aligned.")
        video_embeddings_resampled = video_embeddings
        fmri_data_aligned = fmri_data

    return fmri_data_aligned, video_embeddings_resampled





















# Example usage (can be called from main.py)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Create dummy data
    fmri_len = 100
    voxels = 50
    fmri_tr_test = 1.5
    fmri_dat = np.random.randn(fmri_len, voxels).astype(np.float32)

    video_len_sec = fmri_len * fmri_tr_test # Approx video length
    video_fps_test = 30
    num_frames = int(video_len_sec * video_fps_test)
    chunk_size = 16
    chunk_stride = 8
    num_chunks = (num_frames - chunk_size) // chunk_stride + 1
    embed_dim = 768
    video_embed = np.random.randn(num_chunks, embed_dim).astype(np.float32)

    print(f"Input fMRI: {fmri_dat.shape}")
    print(f"Input Video Embeddings: {video_embed.shape} (for {num_frames} frames at {video_fps_test} FPS)")

    fmri_aligned, video_aligned = align_data(
        preprocess_fmri(fmri_dat),
        preprocess_video_embeddings(video_embed),
        fmri_tr=fmri_tr_test,
        video_fps=video_fps_test,
        num_video_frames_original=num_frames,
        hrf_delay=5.0,
        align_method='shift',
        video_chunk_size=chunk_size,
        video_chunk_stride=chunk_stride
    )

    print(f"Aligned fMRI shape: {fmri_aligned.shape}")
    print(f"Aligned Video shape: {video_aligned.shape}")