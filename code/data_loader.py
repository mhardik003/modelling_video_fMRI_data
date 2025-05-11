import nibabel as nib
import numpy as np
import cv2
import torch
from pathlib import Path
import logging

# Import the map and variant from config
# from config import TRS_DROP_MAP, FMRI_VARIANT # Import FMRI_VARIANT


def load_video_data(video_file_path: Path) -> (list, float):
    """Loads video frames and returns list of frames and FPS."""
    if not video_file_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_file_path}")

    print(f"Loading video data from: {video_file_path}")
    cap = cv2.VideoCapture(str(video_file_path))

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_file_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}, Total frames: {frame_count}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB (OpenCV loads in BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    print(f"Loaded {len(frames)} frames from video.")

    # Simple check if frame count matches expected
    # if frame_count > 0 and len(frames) != frame_count:
    #     logger.warning(f"Frame count mismatch: OpenCV reported {frame_count}, loaded {len(frames)}.")

    return frames, fps





def load_fmri_data(fmri_file_path: Path, stimulus_name: str, expected_tr: float, FMRI_VARIANT, TRS_DROP_MAP) -> np.ndarray:
    """Loads fMRI data, drops stimulus-specific TRs, and returns data array."""
    
    is_srm_data = "srm-recon" in FMRI_VARIANT # Check if 'srm-recon' is in the variant name

    print(f"Loading fMRI data (Variant: {FMRI_VARIANT}, is_srm: {is_srm_data}) for stimulus '{stimulus_name}' from: {fmri_file_path}")

    img = nib.load(fmri_file_path)
    fmri_data = img.get_fdata(dtype=np.float32)

    # Verify TR from header if possible
    try:
        header_tr = img.header.get_zooms()[-1]
        if not np.isclose(header_tr, expected_tr):
            print(f"TR from NIfTI header ({header_tr:.4f}s) differs from expected TR ({expected_tr:.4f}s). Using expected TR.")
    except Exception:
        print("Could not read TR from NIfTI header. Using expected TR.")
        pass

    print(f"Original loaded fMRI shape: {fmri_data.shape}, TR: {expected_tr:.4f}s")

    # Get stimulus-specific drop counts
    drop_start, drop_end = TRS_DROP_MAP[stimulus_name]

    # Handle 4D data (most common NIfTI format)
    if fmri_data.ndim == 4:
        print("Loaded fMRI data is 4D.")
        # NIfTI convention is X, Y, Z, Time. We need to move Time to the first axis.
        # Check header orientation or assume standard X,Y,Z,T -> T,X,Y,Z
        print("Assuming 4th dimension is time. Transposing to Time x Voxels.")
        fmri_data = np.transpose(fmri_data, (3, 0, 1, 2)) # Now T x X x Y x Z
       
        print(f"Shape after ensuring Time is first dim: {fmri_data.shape}")

        # Drop TRs along the time axis (axis 0)
        if drop_start < 0 or drop_end < 0:
             print(f"Invalid drop values ({drop_start}, {drop_end}). Skipping TR dropping.")
        elif fmri_data.shape[0] <= drop_start + drop_end:
             print(f"Cannot drop TRs from data with only {fmri_data.shape[0]} time points.")
             return np.array([], dtype=np.float32)
        else:
            if drop_start > 0:
                fmri_data = fmri_data[drop_start:, ...]
            if drop_end > 0:
                fmri_data = fmri_data[:-drop_end, ...]
            print(f"Dropped TRs ({drop_start} start, {drop_end} end).")

        print(f"Shape after dropping TRs: {fmri_data.shape}")

        
        
        # RESHAPING THE DATA TO FLATTEN ALL THE VOXELS FROM XYZ to one single vector
        
        # If it IS SRM data, it should already represent features.
        # It might be T x F x 1 x 1 or similar. Flatten dimensions 1 onwards.
        if fmri_data.ndim > 2:
                original_dims = fmri_data.shape
                fmri_data = fmri_data.reshape(fmri_data.shape[0], -1) # T x F (Flatten feature dims)
                print(f"Reshaped SRM fMRI from {original_dims} to Time x Features: {fmri_data.shape}")
        else:
                # Should not happen if we checked ndim==2 earlier, but safety check
                print(f"SRM data is already 2D after TR drop. Shape: {fmri_data.shape}")

    else:
        print(f"Loaded fMRI data has unexpected number of dimensions: {fmri_data.ndim}. Shape: {fmri_data.shape}")
        return np.array([], dtype=np.float32)

    return fmri_data