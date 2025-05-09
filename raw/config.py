# src/config.py
import torch
import os
from pathlib import Path
import json # For reading TR
import logging # Add logging import

# --- Assume logger is configured elsewhere or add basic config ---
# Simplest basic config if needed here (better to configure in main.py):
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Get logger instance

# --- Base Paths ---
# set the base dir as "/data/gaurav.bhole/CSAI/Project"
BASE_DIR = Path("/workspace/hardik/") # Adjusted path

DATA_DIR = BASE_DIR / "ds004516-download" # Adjusted path
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- Subject and Stimulus ---
# SUBJECT_ID = "NSD103" # Use format consistent with filenames
TRAIN_SUBJECT_IDS = ["NSD103"] # List of subjects for training
TEST_SUBJECT_ID = "NSD104"     # Single subject for testing
# STIMULI_NAMES = ["iteration", "defeat", "growth", "lemonade"]
STIMULI_NAMES = ["iteration"]

# --- fMRI Parameters ---
PREPROC_DIR = DATA_DIR / "derivatives" / "preprocessed"
FMRI_FILE_TEMPLATE = "sub-{subject_id}_task-{stimulus_lower}_{variant}.nii.gz"
# Raw data directory template (string for formatting later)
RAW_FUNC_DIR_PATH_TEMPLATE = DATA_DIR / "sub-{subject_id}" / "func"
# Raw JSON template (string)
RAW_JSON_FILENAME_TEMPLATE = "sub-{subject_id}_task-{stimulus_lower}_echo-1_bold.json"

# FMRI_VARIANT = "nocensor" # Or "nocensor_srm-recon"
FMRI_VARIANT = "nocensor_srm-recon" 
TR = None # Will be read dynamically
# TRS_DROP_MAP = {
#     "growth": (2, 11),
#     "lemonade": (2, 11),
#     "iteration": (2, 11),
#     "defeat": (2, 12),
# }
TRS_DROP_MAP = {
    "iteration": (2, 11)
}

# --- Video Parameters ---
VIDEO_DIR = DATA_DIR / "stimuli"
VIDEO_FILE_TEMPLATE = "{stimulus_lower}.mp4"
# VIDEO_FILE_TEMPLATE = "{stimulus_lower}_short.mp4"
# VIDEO_EMBEDDING_MODEL = "facebook/timesformer-base-finetuned-k400"
# VIDEO_CHUNK_SIZE = 8
# VIDEO_CHUNK_STRIDE = 4
# VIDEO_EMBEDDING_BATCH_SIZE = 8

# --- Video Encoder Selection ---
# Choose one: 'timesformer', 'videomae', 'xclip'
# This setting will determine which model is loaded and run.
VIDEO_ENCODER_NAME = 'videomae' # CHANGE THIS TO 'videomae' or 'xclip' TO SWITCH
# VIDEO_ENCODER_NAME = 'timesformer' # CHANGE THIS TO 'videomae' or 'xclip' TO SWITCH

VIDEO_MODEL_IDENTIFIERS = {
    'timesformer': "facebook/timesformer-base-finetuned-k400",
    'videomae': "MCG-NJU/videomae-base-finetuned-kinetics", # Using V2
    'xclip': "microsoft/xclip-base-patch32"
}

VIDEO_MODEL_OUTPUT_DIMS = {
    'timesformer': 768,
    'videomae': 768,
    'xclip': 512 # X-Clip base has 512 dim output for vision/text
}

# --- Dynamically Set Based on Selection (will be updated in main.py) ---
# Placeholder - these get overwritten at runtime based on VIDEO_ENCODER_NAME
VIDEO_EMBEDDING_MODEL = None # Will be set from VIDEO_MODEL_IDENTIFIERS
DEC_OUTPUT_DIM = None        # Will be set from VIDEO_MODEL_OUTPUT_DIMS

# --- Video Processing Parameters (might need slight tuning per model if issues arise) ---
VIDEO_CHUNK_SIZE = 8 # Frames per model input clip (TimeSformer/VideoMAE usually 8 or 16)
VIDEO_CHUNK_STRIDE = 4 # Overlap between chunks
VIDEO_EMBEDDING_BATCH_SIZE = 8 # Reduce if OOM during *embedding extraction*

# --- Preprocessing Parameters ---
ALIGN_METHOD = "shift"
HRF_DELAY = 4.0
DO_FMRI_ZSCORE = True
DO_VIDEO_ZSCORE = True

# # --- Model Parameters ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ENC_HIDDEN_DIM = 1024
# ENC_OUTPUT_DIM = -1
# DEC_INPUT_DIM = -1
# DEC_HIDDEN_DIM = 2048
# DEC_OUTPUT_DIM = 768

# --- Model Parameters (MLP - Non-Temporal) ---
USE_TEMPORAL_MODELS = False # Ensure this is False to use MLP
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# --- MLP Hidden Dims ---
ENC_HIDDEN_DIM = 1024 # Hidden dim for MLP encoding model
DEC_HIDDEN_DIM = 1024 # Hidden dim for MLP decoding model (reduced from 2048)
# --- Input/Output Dims (set dynamically) ---
ENC_OUTPUT_DIM = -1   # Set dynamically (PCA components)
DEC_INPUT_DIM = -1    # Set dynamically (PCA components)

APPLY_PCA = False # Flag to enable/disable PCA
PCA_N_COMPONENTS = 500 # Target number of fMRI features after PCA

# --- Training Parameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
# EPOCHS = 10

# --- Evaluation ---
EVAL_METRICS = ['mse', 'pearsonr', 'r2']

# --- Helper to read TR ---
def get_tr_from_json(subject_id: str, stimulus_lower: str) -> float:
    """Reads RepetitionTime from the BOLD JSON sidecar file."""
    try:
        # Construct the path to the func directory using f-string within Path parts
        func_dir_path = DATA_DIR / f"sub-{subject_id}" / "func"

        # Construct the specific json filename using f-string
        json_filename = f"sub-{subject_id}_task-{stimulus_lower}_echo-1_bold.json"

        # Combine directory and filename to get the full path
        json_path = func_dir_path / json_filename

        logger.debug(f"Attempting to read TR from: {json_path}")

        if not json_path.exists():
            logger.warning(f"Specific JSON file not found: {json_path}. Searching fallback in {func_dir_path}")
            # Fallback: Try finding *any* json in the func dir for the TR
            json_files = list(func_dir_path.glob('sub-*.json')) # More specific glob pattern
            if not json_files:
                 raise FileNotFoundError(f"No JSON sidecar found in {func_dir_path} to determine TR.")
            json_path = json_files[0] # Use the first one found
            logger.warning(f"Using fallback JSON for TR: {json_path.name}")

        logger.debug(f"Reading metadata from: {json_path}")
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        if 'RepetitionTime' not in metadata:
             raise ValueError(f"'RepetitionTime' key not found in JSON file: {json_path}")

        tr = float(metadata['RepetitionTime'])
        logger.debug(f"Successfully read TR={tr} from {json_path.name}")
        return tr

    except FileNotFoundError as e:
        # Re-raise with more context maybe
        logger.error(f"File not found during TR lookup: {e}")
        raise e
    except ValueError as e:
        logger.error(f"Value error during TR lookup: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error reading or parsing JSON file {json_path}: {e}", exc_info=True)
        # Wrap unexpected errors for clarity
        raise IOError(f"Error reading or parsing JSON file {json_path}: {e}")

# --- Potential Future Extensions ---
# - Incorporate audio features from videos.
# - Use text features from stimuli/*.txt (e.g., character presence, event annotations).
# - Explore advanced alignment models (e.g., dynamic time warping variants, attention-based).
# - Implement HRF convolution properly in preprocessing.py.