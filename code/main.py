# # src/main.py
# import logging
# import time
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import pickle
# import matplotlib.pyplot as plt

# # Import project modules
# import config # Import base config
# from config import get_tr_from_json # Import the helper function
# from data_loader import load_fmri_data, load_video_data
# from video_encoder import VideoFeatureExtractor
# from preprocessing import preprocess_fmri, preprocess_video_embeddings, align_data
# # Import specific models needed (MLP in this case)
# from models import EncodingModel, DecodingModel
# from train import prepare_dataloaders, run_training
# from evaluate import evaluate_model, plot_predictions
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__)

# def save_data(data, filename: Path):
#     logger.info(f"Saving data to {filename}")
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# def load_data(filename: Path):
#     logger.info(f"Loading data from {filename}")
#     if not filename.exists():
#         logger.error(f"Cache file not found: {filename}")
#         return None
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
    
# def process_subject_data(subject_id, fmri_tr, video_extractor, chosen_encoder):
#     """Loads, preprocesses, and aligns data for a single subject."""
#     logger.info(f"--- Processing data for Subject: {subject_id} ---")
#     subject_fmri_aligned = []
#     subject_video_aligned = []

#     for stimulus_lower in config.STIMULI_NAMES:
#         # --- Define file paths ---
#         # ... (Path logic using subject_id and stimulus_lower) ...
#         fmri_filename = config.FMRI_FILE_TEMPLATE.format(
#             subject_id=subject_id, # Use passed subject_id
#             stimulus_lower=stimulus_lower,
#             variant=config.FMRI_VARIANT
#         )
#         fmri_path = config.PREPROC_DIR / f"sub-{subject_id}" / fmri_filename # Use passed subject_id
#         # ... (Video path logic remains the same) ...
#         video_filename = config.VIDEO_FILE_TEMPLATE.format(stimulus_lower=stimulus_lower)
#         video_path = config.VIDEO_DIR / video_filename

#         # --- Cache Keys (Include Subject ID) ---
#         safe_encoder_name = chosen_encoder.replace('/','_')
#         aligned_cache_key = f"sub-{subject_id}_{stimulus_lower}_{config.FMRI_VARIANT}_{safe_encoder_name}_aligned.pkl" # Add subject_id
#         aligned_cache_file = config.CACHE_DIR / aligned_cache_key
#         embedding_cache_key = f"{stimulus_lower}_{safe_encoder_name}_embeddings.npy" # Embeddings are stimulus-specific, not subject-specific
#         embedding_cache_file = config.CACHE_DIR / embedding_cache_key

#         # --- Load/Process Aligned Data (Logic mostly the same as before) ---
#         if aligned_cache_file.exists():
#             # ... (Load from cache) ...
#             cached_data = load_data(aligned_cache_file)
#             if cached_data: # Add validation checks here
#                  fmri_aligned, video_aligned = cached_data
#                  expected_vid_dim = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
#                  if fmri_aligned.ndim == 2 and video_aligned.ndim == 2 and fmri_aligned.shape[0] == video_aligned.shape[0] and video_aligned.shape[1] == expected_vid_dim:
#                     subject_fmri_aligned.append(fmri_aligned)
#                     subject_video_aligned.append(video_aligned)
#                     logger.info(f"Loaded cached aligned data shapes for {subject_id}/{stimulus_lower}: fMRI {fmri_aligned.shape}, Video {video_aligned.shape}")
#                     continue
#                  else: logger.warning(f"Invalid cached data for {subject_id}/{stimulus_lower}. Recomputing.")

#         # --- Load Raw Data ---
#         try:
#             # ... (Load fmri_data using subject_id, stimulus_lower, fmri_tr) ...
#             fmri_data = load_fmri_data(fmri_path, stimulus_lower, fmri_tr)
#             if fmri_data.size == 0: continue
#             # Video loading is stimulus specific, maybe cache frames? For now, reload.
#             video_frames, video_fps = load_video_data(video_path)
#             if not video_frames: continue
#             num_video_frames_original = len(video_frames)
#         except Exception as e:
#             logger.error(f"Error loading data for {subject_id}/{stimulus_lower}: {e}.", exc_info=True)
#             continue

#         # --- Video Embeddings (Load from cache or extract) ---
#         # ... (Existing logic using embedding_cache_file and video_extractor) ...
#         # Need to ensure video_embeddings variable is correctly populated or extracted

#         # --- Preprocess & Align ---
#         # ... (Existing logic using fmri_data, video_embeddings, fmri_tr etc.) ...
#         # Save aligned data to subject-specific cache file
#         try:
#             # ... (Call preprocess_fmri, preprocess_video_embeddings, align_data) ...
#             fmri_processed = preprocess_fmri(fmri_data)
#             video_embeddings_processed = preprocess_video_embeddings(video_embeddings) # Make sure video_embeddings exists
#             fmri_aligned, video_aligned = align_data(...) # Pass correct args
#             if fmri_aligned.shape[0] == 0 or video_aligned.shape[0] == 0: continue
#             save_data((fmri_aligned, video_aligned), aligned_cache_file)
#             subject_fmri_aligned.append(fmri_aligned)
#             subject_video_aligned.append(video_aligned)
#         except Exception as e:
#             logger.error(f"Error during preprocess/align for {subject_id}/{stimulus_lower}: {e}", exc_info=True)
#             continue

#     # --- Concatenate data for *this* subject ---
#     if not subject_fmri_aligned:
#         logger.warning(f"No data processed for subject {subject_id}.")
#         return None, None

#     final_fmri = np.concatenate(subject_fmri_aligned, axis=0)
#     final_video = np.concatenate(subject_video_aligned, axis=0)
#     logger.info(f"--- Concatenated Data Shapes for Subject {subject_id} ---")
#     logger.info(f"Subject fMRI data: {final_fmri.shape}")
#     logger.info(f"Subject Video embeddings: {final_video.shape}")
#     return final_fmri, final_video

    
# def main():
#     logger.info("--- Starting fMRI-Video Alignment Pipeline ---")
    
#     # --- Determine and Set Video Encoder Configuration ---
#     chosen_encoder = config.VIDEO_ENCODER_NAME
#     if chosen_encoder not in config.VIDEO_MODEL_IDENTIFIERS:
#         logger.error(f"Invalid VIDEO_ENCODER_NAME '{chosen_encoder}' in config. Choose from {list(config.VIDEO_MODEL_IDENTIFIERS.keys())}.")
#         return
#     config.VIDEO_EMBEDDING_MODEL = config.VIDEO_MODEL_IDENTIFIERS[chosen_encoder]
#     config.DEC_OUTPUT_DIM = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
#     logger.info(f"Selected Video Encoder: {chosen_encoder} (ID: {config.VIDEO_EMBEDDING_MODEL}, Dim: {config.DEC_OUTPUT_DIM})")
    
#     logger.info(f"Configuration: Subject=sub-{config.SUBJECT_ID}, Device={config.DEVICE}")
#     logger.info(f"Using fMRI variant: {config.FMRI_VARIANT}")

#     # --- Determine TR for the subject ---
#     try:
#         # Read TR using the first stimulus as reference (assuming constant TR)
#         reference_stimulus = config.STIMULI_NAMES[0]
#         fmri_tr = get_tr_from_json(config.SUBJECT_ID, reference_stimulus)
#         logger.info(f"Determined TR for Subject {config.SUBJECT_ID}: {fmri_tr:.4f}s")
#         # Assign to config if needed elsewhere, though passing it is safer
#         config.TR = fmri_tr
#     except (FileNotFoundError, ValueError, IOError) as e:
#         logger.error(f"Failed to determine TR for subject {config.SUBJECT_ID}: {e}. Exiting.")
#         return

#     # Initialize Video Feature Extractor with the chosen model identifier
#     try:
#         video_extractor = VideoFeatureExtractor(
#             model_identifier=config.VIDEO_EMBEDDING_MODEL,
#             device=config.DEVICE
#         )
#     except Exception as e:
#         # Log the actual exception caught in main.py
#         logger.error(f"Failed to initialize VideoFeatureExtractor in main: {e}", exc_info=True)
#         return # Exit after logging the detailed error

#     all_fmri_aligned = []
#     all_video_aligned = []

#     # --- Process each stimulus ---
#     for stimulus_lower in config.STIMULI_NAMES:
#         logger.info(f"--- Processing Stimulus: {stimulus_lower} ---")

#         # --- Define file paths ---
#         fmri_filename = config.FMRI_FILE_TEMPLATE.format(
#             subject_id=config.SUBJECT_ID,
#             stimulus_lower=stimulus_lower,
#             variant=config.FMRI_VARIANT
#         )
#         # Correct path construction for derivatives
#         fmri_path = config.PREPROC_DIR / f"sub-{config.SUBJECT_ID}" / fmri_filename

#         video_filename = config.VIDEO_FILE_TEMPLATE.format(stimulus_lower=stimulus_lower)
#         video_path = config.VIDEO_DIR / video_filename
        
#         # --- Cache Keys (MUST include video encoder name) ---
#         # --- Check cache for aligned data ---
#         # Include TR and alignment method in cache key? Maybe not essential if config is stable.
#         safe_encoder_name = chosen_encoder.replace('/','_') # Make sure name is filename-safe
#         aligned_cache_key = f"sub-{config.SUBJECT_ID}_{stimulus_lower}_{config.FMRI_VARIANT}_{safe_encoder_name}_aligned.pkl"
#         aligned_cache_file = config.CACHE_DIR / aligned_cache_key
#         embedding_cache_key = f"{stimulus_lower}_{safe_encoder_name}_embeddings.npy"
#         embedding_cache_file = config.CACHE_DIR / embedding_cache_key
#         # --- End Cache Keys ---

#         # cache_key = f"sub-{config.SUBJECT_ID}_{stimulus_lower}_{config.FMRI_VARIANT}_{config.VIDEO_EMBEDDING_MODEL.replace('/','_')}_aligned.pkl"
#         # aligned_cache_file = config.CACHE_DIR / cache_key
#         if aligned_cache_file.exists():
#             logger.info(f"Found cached aligned data for {stimulus_lower} ({chosen_encoder}): {aligned_cache_file}")
#             cached_data = load_data(aligned_cache_file)
#             if cached_data:
#                 fmri_aligned, video_aligned = cached_data
#                 expected_vid_dim = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
#                 # Basic check if dimensions seem plausible (optional)
#                 if fmri_aligned.ndim == 2 and video_aligned.ndim == 2 and fmri_aligned.shape[0] == video_aligned.shape[0]:
#                      all_fmri_aligned.append(fmri_aligned)
#                      all_video_aligned.append(video_aligned)
#                      logger.info(f"Loaded aligned data shapes: fMRI {fmri_aligned.shape}, Video {video_aligned.shape}")
#                      continue # Skip processing if cache loaded successfully
#                 else:
#                      logger.warning(f"Cached data {aligned_cache_file} has unexpected shape/dim ({video_aligned.shape[1]} vs {expected_vid_dim}) for {chosen_encoder}. Recomputing.")


#         # --- Load Data ---
#         try:
#             start_load = time.time()
#             # Pass stimulus name and determined TR to loader
#             fmri_data = load_fmri_data(fmri_path, stimulus_lower, fmri_tr)
#             if fmri_data.size == 0:
#                  logger.error(f"fMRI data loading failed or resulted in empty array for {stimulus_lower}. Skipping stimulus.")
#                  continue

#             video_frames, video_fps = load_video_data(video_path)
#             logger.info(f"Data loading took {time.time() - start_load:.2f}s")

#             if not video_frames:
#                 logger.error(f"No frames loaded for video: {video_path}. Skipping stimulus.")
#                 continue
#             num_video_frames_original = len(video_frames)

#         except FileNotFoundError as e:
#             logger.error(f"Error loading data for {stimulus_lower}: {e}. Skipping stimulus.")
#             continue
#         except Exception as e:
#             logger.error(f"An unexpected error occurred during data loading for {stimulus_lower}: {e}", exc_info=True)
#             continue

#         # --- Extract Video Embeddings (or load from cache) ---
#         embedding_cache_key = f"{stimulus_lower}_{config.VIDEO_EMBEDDING_MODEL.replace('/','_')}_embeddings.npy"
#         embedding_cache_file = config.CACHE_DIR / embedding_cache_key

#         # Extract Video Embeddings (or load from cache)
#         if embedding_cache_file.exists():
#              logger.info(f"Loading cached {chosen_encoder} video embeddings from: {embedding_cache_file}")
#              video_embeddings = np.load(embedding_cache_file)
#              # Basic check on embedding dimension
#              expected_vid_dim = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
#              if video_embeddings.ndim != 2 or video_embeddings.shape[1] != expected_vid_dim:
#                   logger.warning(f"Cached embeddings {embedding_cache_file} have wrong shape/dim ({video_embeddings.shape}) for {chosen_encoder}. Re-extracting.")
#                   embedding_cache_file.unlink() # Delete invalid cache file
#                   video_embeddings = None # Force re-extraction
#              else:
#                  logger.info(f"Loaded {chosen_encoder} embeddings shape: {video_embeddings.shape}")

#         else:
#              video_embeddings = None # Ensure it's None if file doesn't exist

#         if video_embeddings is None: # Condition to extract
#              logger.info(f"Extracting {chosen_encoder} video embeddings...")
#              start_embed = time.time()
#              try:
#                  video_embeddings = video_extractor.extract_features(video_frames, batch_size=config.VIDEO_EMBEDDING_BATCH_SIZE)
#                  logger.info(f"Video embedding extraction took {time.time() - start_embed:.2f}s")
#                  if video_embeddings.size == 0: raise ValueError("Extractor returned empty embeddings")
#                  # Dimension check after extraction
#                  expected_vid_dim = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
#                  if video_embeddings.shape[1] != expected_vid_dim:
#                       logger.error(f"Extracted {chosen_encoder} embeddings have wrong dim! Expected {expected_vid_dim}, Got {video_embeddings.shape[1]}. Check model/config.")
#                       # Decide how to handle: continue run, exit, skip stimulus?
#                       continue # Skip this stimulus
#                  np.save(embedding_cache_file, video_embeddings)
#                  logger.info(f"Saved {chosen_encoder} video embeddings to: {embedding_cache_file}")
#              except Exception as e:
#                  logger.error(f"Video embedding extraction failed for {chosen_encoder}: {e}", exc_info=True)
#                  continue # Skip this stimulus


#         # --- Preprocess & Align ---
#         try:
#             start_preprocess = time.time()
#             fmri_processed = preprocess_fmri(fmri_data)
#             video_embeddings_processed = preprocess_video_embeddings(video_embeddings)

#             # Pass the dynamically determined fmri_tr here
#             fmri_aligned, video_aligned = align_data(
#                 fmri_processed,
#                 video_embeddings_processed,
#                 fmri_tr=fmri_tr, # Use the TR read earlier
#                 video_fps=video_fps,
#                 num_video_frames_original=num_video_frames_original,
#                 hrf_delay=config.HRF_DELAY,
#                 align_method=config.ALIGN_METHOD,
#                 video_chunk_size=video_extractor.num_frames_per_clip, # Use actual chunk size from extractor
#                 video_chunk_stride=config.VIDEO_CHUNK_STRIDE
#             )
#             logger.info(f"Preprocessing and alignment took {time.time() - start_preprocess:.2f}s")

#             if fmri_aligned.shape[0] == 0 or video_aligned.shape[0] == 0:
#                 logger.warning(f"Alignment resulted in empty data for {stimulus_lower}. Skipping stimulus.")
#                 continue

#             # Cache the aligned data for this stimulus
#             save_data((fmri_aligned, video_aligned), aligned_cache_file)
#             logger.info(f"Saved aligned data shapes: fMRI {fmri_aligned.shape}, Video {video_aligned.shape}")


#             all_fmri_aligned.append(fmri_aligned)
#             all_video_aligned.append(video_aligned)

#         except Exception as e:
#             logger.error(f"An error occurred during preprocessing/alignment for {stimulus_lower}: {e}", exc_info=True)
#             continue


#     # --- Concatenate, Train, Evaluate (after loop) ---
#     if not all_fmri_aligned or not all_video_aligned:
#         logger.error("No usable aligned data generated from any stimulus. Exiting.")
#         return

#     # --- Consistency Check before Concatenation ---
#     try:
#         fmri_dim_raw = all_fmri_aligned[0].shape[1]
#         video_embed_dim = all_video_aligned[0].shape[1]
#         if not all(d.shape[1] == fmri_dim_raw for d in all_fmri_aligned): # Check raw fMRI dim
#             logger.error("Mismatch in raw fMRI dimensions across stimuli.")
#             return
#         if not all(d.shape[1] == video_embed_dim for d in all_video_aligned):
#             logger.error(f"Mismatch in video embedding dimensions ({video_embed_dim} vs expected {config.DEC_OUTPUT_DIM}).")
#             return
#         # Verify video dim matches expectation
#         if video_embed_dim != config.DEC_OUTPUT_DIM:
#              logger.error(f"Concatenated video dim ({video_embed_dim}) does not match expected for {chosen_encoder} ({config.DEC_OUTPUT_DIM}). Check config/cache.")
#              return
#     except IndexError:
#          logger.error("Cannot perform consistency check, no aligned data collected.")
#          return

#     # Concatenate data
#     final_fmri_data_raw = np.concatenate(all_fmri_aligned, axis=0)
#     final_video_embeddings = np.concatenate(all_video_aligned, axis=0)
#     logger.info(f"--- Concatenated Raw Data Shapes ---")
#     logger.info(f"Concatenated fMRI data (Raw/SRM): {final_fmri_data_raw.shape}")
#     logger.info(f"Concatenated Video embeddings ({chosen_encoder}): {final_video_embeddings.shape}")

#     # --- Apply PCA (Optional but Recommended) ---
#     if config.APPLY_PCA:
#         logger.info(f"Applying PCA to fMRI data. Target components: {config.PCA_N_COMPONENTS}")
#         if final_fmri_data_raw.shape[1] <= config.PCA_N_COMPONENTS:
#              logger.warning(f"Number of features ({final_fmri_data_raw.shape[1]}) is less than or equal to target PCA components ({config.PCA_N_COMPONENTS}). Skipping PCA.")
#              final_fmri_data = final_fmri_data_raw
#         else:
#             # 1. Scale data before PCA
#             start_pca = time.time()
#             scaler = StandardScaler()
#             fmri_scaled = scaler.fit_transform(final_fmri_data_raw)
#             pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=42)
#             final_fmri_data = pca.fit_transform(fmri_scaled)
#             logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
#             logger.info(f"PCA finished in {time.time() - start_pca:.2f}s.")

#             # TODO: Save the PCA model and scaler if you need to inverse transform later
#             # pca_model_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_pca_model.pkl"
#             # scaler_model_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_scaler_model.pkl"
#             # with open(pca_model_path, 'wb') as f: pickle.dump(pca, f)
#             # with open(scaler_model_path, 'wb') as f: pickle.dump(scaler, f)
#     else:
#          logger.info("Skipping PCA based on config.")
#          final_fmri_data = final_fmri_data_raw # Use the raw SRM data


#     logger.info(f"--- Final Data Shapes for Training ---")
#     logger.info(f"Final fMRI data: {final_fmri_data.shape}") # This will now be (Time, PCA_N_COMPONENTS) if PCA applied
#     logger.info(f"Final Video embeddings: {final_video_embeddings.shape}")


#     # --- Prepare DataLoaders ---
#     # Prepare DataLoaders (ensure USE_TEMPORAL_MODELS is False in config for MLP)
#     train_loader, val_loader, test_loader = prepare_dataloaders(
#         final_fmri_data, final_video_embeddings,
#         batch_size=config.BATCH_SIZE,
#         )
#     if train_loader is None: return

#     # Update config dims (important!)
#     config.ENC_OUTPUT_DIM = final_fmri_data.shape[1] # PCA dim or raw SRM dim
#     config.DEC_INPUT_DIM = final_fmri_data.shape[1]  # PCA dim or raw SRM dim

#     # # --- Concatenate data ---
#     # final_fmri_data = np.concatenate(all_fmri_aligned, axis=0)
#     # final_video_embeddings = np.concatenate(all_video_aligned, axis=0)
#     # logger.info(f"--- Concatenated Data Shapes ---")
#     # logger.info(f"Final fMRI data: {final_fmri_data.shape}")
#     # logger.info(f"Final Video embeddings: {final_video_embeddings.shape}")

#     # # --- Prepare DataLoaders ---
#     # train_loader, val_loader, test_loader = prepare_dataloaders(final_fmri_data, final_video_embeddings)

#     # Update config with actual dimensions
#     config.ENC_OUTPUT_DIM = final_fmri_data.shape[1]
#     config.DEC_INPUT_DIM = final_fmri_data.shape[1]
#     # Ensure video embedding dim matches model expectation
#     if final_video_embeddings.shape[1] != config.DEC_OUTPUT_DIM:
#          logger.warning(f"Actual video embedding dimension ({final_video_embeddings.shape[1]}) differs from config.DEC_OUTPUT_DIM ({config.DEC_OUTPUT_DIM}). Using actual dimension.")
#          config.DEC_OUTPUT_DIM = final_video_embeddings.shape[1]
         
#     # # --- Model Initialization (MLP Models) ---
#     # if config.USE_TEMPORAL_MODELS:
#     #     logger.error("Config mismatch: USE_TEMPORAL_MODELS is True, but MLP logic is expected. Set it to False.")
#     #     return

#     logger.info("Initializing MLP Models...")
#     encoding_model = EncodingModel(
#         video_embed_dim=config.DEC_OUTPUT_DIM, # Correct video dim
#         fmri_dim=config.ENC_OUTPUT_DIM,        # Correct fMRI (PCA) dim
#         hidden_dim=config.ENC_HIDDEN_DIM
#     )
#     decoding_model = DecodingModel(
#         fmri_dim=config.DEC_INPUT_DIM,         # Correct fMRI (PCA) dim
#         video_embed_dim=config.DEC_OUTPUT_DIM, # Correct video dim
#         hidden_dim=config.DEC_HIDDEN_DIM
#     )

#     # --- Encoding Model Training & Evaluation ---
#     logger.info(f"\n--- Training Encoding Model (MLP, Video: {chosen_encoder}) ---")
#     encoding_model = run_training(
#         encoding_model, train_loader, val_loader, config.DEVICE,
#         task='encoding', epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE,
#         )

#     logger.info(f"\n--- Evaluating Encoding Model (MLP, Video: {chosen_encoder}) ---")
#     enc_results, enc_targets, enc_predictions = evaluate_model(
#          encoding_model, test_loader, nn.MSELoss(), config.DEVICE,
#          task='encoding'
#          )

#     # --- Add encoder name to output files ---
#     enc_model_suffix = f"_mlp_{chosen_encoder}"
#     enc_model_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_encoding_model{enc_model_suffix}.pt"
#     torch.save(encoding_model.state_dict(), enc_model_path)
#     logger.info(f"Saved Encoding model to {enc_model_path}")

#     enc_fig = plot_predictions(enc_targets, enc_predictions, n_samples=5, title=f"Encoding (MLP, {chosen_encoder}): Target vs Predicted fMRI PCA (Test Set) - Subj {config.SUBJECT_ID}")
#     enc_plot_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_encoding_predictions{enc_model_suffix}.png"
#     enc_fig.savefig(enc_plot_path)
#     plt.close(enc_fig)
#     logger.info(f"Saved Encoding prediction plot to {enc_plot_path}")

#     # --- Decoding Model Training & Evaluation ---
#     logger.info(f"\n--- Training Decoding Model (MLP, Video: {chosen_encoder}) ---")
#     if config.DEVICE == 'cuda':
#         del encoding_model
#         if 'enc_targets' in locals(): del enc_targets
#         if 'enc_predictions' in locals(): del enc_predictions
#         torch.cuda.empty_cache()
#         logger.info("Cleared CUDA cache before starting decoding model training.")

#     decoding_model = run_training(
#         decoding_model, train_loader, val_loader, config.DEVICE,
#         task='decoding', epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE
#         )

#     logger.info(f"\n--- Evaluating Decoding Model (MLP, Video: {chosen_encoder}) ---")
#     dec_results, dec_targets, dec_predictions = evaluate_model(
#         decoding_model, test_loader, nn.MSELoss(), config.DEVICE,
#         task='decoding'
#         )

#     dec_model_suffix = f"_mlp_{chosen_encoder}"
#     dec_model_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_decoding_model{dec_model_suffix}.pt"
#     torch.save(decoding_model.state_dict(), dec_model_path)
#     logger.info(f"Saved Decoding model to {dec_model_path}")

#     dec_fig = plot_predictions(dec_targets, dec_predictions, n_samples=5, title=f"Decoding (MLP, {chosen_encoder}): Target vs Predicted Video ({chosen_encoder}) Emb (Test Set) - Subj {config.SUBJECT_ID}")
#     dec_plot_path = config.OUTPUT_DIR / f"sub-{config.SUBJECT_ID}_decoding_predictions{dec_model_suffix}.png"
#     dec_fig.savefig(dec_plot_path)
#     plt.close(dec_fig)
#     logger.info(f"Saved Decoding prediction plot to {dec_plot_path}")

#     logger.info("\n--- Pipeline Finished ---")


# if __name__ == "__main__":
#     main()

# src_working/main.py
import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn # For criterion hint
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt

# Import project modules
import config # Import base config
from config import get_tr_from_json
from data_loader import load_fmri_data, load_video_data
from video_encoder import VideoFeatureExtractor
from preprocessing import preprocess_fmri, preprocess_video_embeddings, align_data
from models import EncodingModel, DecodingModel, VoxelWiseEncodingModel # Using MLP models
from torch.utils.data import Dataset, TensorDataset, DataLoader 
# Make sure train.py has the correct prepare_dataloaders and a training function without validation
from train import train_epoch # Import reverted train_epoch
from evaluate import evaluate_model, plot_predictions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def save_data(data, filename: Path):
    logger.info(f"Saving data to {filename}")
    filename.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename: Path):
    logger.info(f"Loading data from {filename}")
    if not filename.exists():
        logger.error(f"Cache file not found: {filename}")
        return None
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading cache file {filename}: {e}", exc_info=True)
        return None

def process_subject_data(subject_id, fmri_tr, video_extractor, chosen_encoder):
    """Loads, preprocesses, and aligns data for a single subject."""
    logger.info(f"--- Processing data for Subject: {subject_id} ---")
    subject_fmri_aligned = []
    subject_video_aligned = []
    # --- Determine Expected Video Dimension ---
    if chosen_encoder not in config.VIDEO_MODEL_OUTPUT_DIMS:
         logger.error(f"Cannot find output dimension for encoder '{chosen_encoder}' in config.VIDEO_MODEL_OUTPUT_DIMS.")
         return None, None # Cannot proceed without knowing expected dim
    expected_vid_dim = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
    safe_encoder_name = chosen_encoder.replace('/','_') # Safe name for filenames

    video_embeddings_cache = {} # Cache loaded video embeddings per stimulus

    for stimulus_lower in config.STIMULI_NAMES:
        logger.info(f"Processing {subject_id} / {stimulus_lower}...")
        # --- Define file paths ---
        fmri_filename = config.FMRI_FILE_TEMPLATE.format(
            subject_id=subject_id,
            stimulus_lower=stimulus_lower,
            variant=config.FMRI_VARIANT
        )
        fmri_path = config.PREPROC_DIR / f"sub-{subject_id}" / fmri_filename
        video_filename = config.VIDEO_FILE_TEMPLATE.format(stimulus_lower=stimulus_lower)
        video_path = config.VIDEO_DIR / video_filename

        # --- Cache Keys ---
        aligned_cache_key = f"sub-{subject_id}_{stimulus_lower}_{config.FMRI_VARIANT}_{safe_encoder_name}_aligned.pkl"
        aligned_cache_file = config.CACHE_DIR / aligned_cache_key
        # Video embeddings are stimulus-specific, cache key doesn't need subject_id
        embedding_cache_key = f"{stimulus_lower}_{safe_encoder_name}_embeddings.npy"
        embedding_cache_file = config.CACHE_DIR / embedding_cache_key

        # --- Check cache for ALIGNED data first ---
        if aligned_cache_file.exists():
            cached_data = load_data(aligned_cache_file)
            if cached_data:
                fmri_aligned, video_aligned = cached_data
                if fmri_aligned.ndim == 2 and video_aligned.ndim == 2 and \
                   fmri_aligned.shape[0] == video_aligned.shape[0] and \
                   video_aligned.shape[1] == expected_vid_dim:
                    subject_fmri_aligned.append(fmri_aligned)
                    subject_video_aligned.append(video_aligned)
                    logger.info(f"Loaded cached aligned data for {subject_id}/{stimulus_lower}")
                    continue # Go to next stimulus if aligned cache is valid
                else:
                    logger.warning(f"Invalid cached aligned data shape/dim for {subject_id}/{stimulus_lower}. Recomputing.")

        # --- Load Raw fMRI Data ---
        try:
            start_load = time.time()
            logger.info(f"Loading fMRI from: {fmri_path}")
            fmri_data = load_fmri_data(fmri_path, stimulus_lower, fmri_tr)
            if fmri_data is None or fmri_data.size == 0:
                 logger.warning(f"fMRI data loading failed or resulted in empty array for {subject_id}/{stimulus_lower}.")
                 continue # Skip this stimulus
            logger.info(f"fMRI loaded in {time.time() - start_load:.2f}s. Shape: {fmri_data.shape}")
        except FileNotFoundError:
            logger.warning(f"fMRI file not found for {subject_id}/{stimulus_lower}: {fmri_path}. Skipping stimulus.")
            continue
        except Exception as e:
            logger.error(f"Error loading fMRI for {subject_id}/{stimulus_lower}: {e}", exc_info=True)
            continue

        # --- Load/Cache Video Embeddings (Stimulus Specific) ---
        video_embeddings = None
        if stimulus_lower in video_embeddings_cache:
            video_embeddings = video_embeddings_cache[stimulus_lower]
            logger.info(f"Using pre-loaded video embeddings for {stimulus_lower}.")
        elif embedding_cache_file.exists():
            logger.info(f"Loading cached {chosen_encoder} video embeddings from: {embedding_cache_file}")
            video_embeddings = np.load(embedding_cache_file)
            if video_embeddings.ndim != 2 or video_embeddings.shape[1] != expected_vid_dim:
                logger.warning(f"Cached embeddings {embedding_cache_file} have wrong shape/dim ({video_embeddings.shape}). Re-extracting.")
                embedding_cache_file.unlink()
                video_embeddings = None
            else:
                video_embeddings_cache[stimulus_lower] = video_embeddings # Store in memory cache
        else:
            video_embeddings = None # Needs extraction

        # --- Extract Video Embeddings if needed ---
        if video_embeddings is None:
            try:
                # Load video frames (only if extraction is needed)
                logger.info(f"Loading video frames from: {video_path}")
                start_vid_load = time.time()
                video_frames, video_fps_load_check = load_video_data(video_path) # Get FPS here too
                if not video_frames:
                     logger.error(f"Failed to load video frames for {stimulus_lower}. Skipping.")
                     continue
                logger.info(f"Video frames loaded in {time.time() - start_vid_load:.2f}s.")

                logger.info(f"Extracting {chosen_encoder} video embeddings for {stimulus_lower}...")
                start_embed = time.time()
                video_embeddings = video_extractor.extract_features(video_frames, batch_size=config.VIDEO_EMBEDDING_BATCH_SIZE)
                logger.info(f"Video embedding extraction took {time.time() - start_embed:.2f}s")

                if video_embeddings is None or video_embeddings.size == 0: raise ValueError("Extractor returned empty embeddings")
                if video_embeddings.shape[1] != expected_vid_dim:
                     logger.error(f"Extracted {chosen_encoder} dim {video_embeddings.shape[1]} != expected {expected_vid_dim}.")
                     continue # Skip if extraction yields wrong dimension

                np.save(embedding_cache_file, video_embeddings)
                logger.info(f"Saved {chosen_encoder} video embeddings to: {embedding_cache_file}")
                video_embeddings_cache[stimulus_lower] = video_embeddings # Store in memory cache
                # Need video_fps for alignment, get it from the load_video_data call
                video_fps = video_fps_load_check
                num_video_frames_original = len(video_frames)

            except Exception as e:
                logger.error(f"Video embedding extraction failed for {stimulus_lower} / {chosen_encoder}: {e}", exc_info=True)
                continue
        else:
             # If embeddings loaded from cache, we still need FPS and frame count for alignment
             # Re-load video just to get metadata if not already loaded - less efficient
             # A better way would be to cache metadata alongside embeddings
             try:
                 # Quick load just for metadata
                 _ , video_fps_check = load_video_data(video_path)
                 # Assuming frame count isn't strictly needed if embeddings exist, but FPS is
                 video_fps = video_fps_check
                 # We don't have exact num_frames_original if only cache was loaded,
                 # but maybe it's not strictly needed by align_data if using chunk times? Check align_data.
                 # For safety, let's pass 0 or estimate if needed by align_data.
                 # Revisit align_data's use of num_video_frames_original. If it's just for logging, it's ok.
                 num_video_frames_original = 0 # Placeholder if only cache loaded
             except Exception as e:
                  logger.error(f"Could not load video metadata for {video_path} even though embeddings exist: {e}. Skipping alignment.")
                  continue


        # --- Preprocess & Align ---
        try:
            start_preprocess = time.time()
            fmri_processed = preprocess_fmri(fmri_data)
            video_embeddings_processed = preprocess_video_embeddings(video_embeddings)
            fmri_aligned, video_aligned = align_data(
                fmri_processed,
                video_embeddings_processed,
                fmri_tr=fmri_tr,
                video_fps=video_fps, # Use FPS obtained above
                num_video_frames_original=num_video_frames_original, # Use value obtained above
                hrf_delay=config.HRF_DELAY,
                align_method=config.ALIGN_METHOD,
                video_chunk_size=video_extractor.num_frames_per_clip,
                video_chunk_stride=config.VIDEO_CHUNK_STRIDE
            )
            logger.info(f"Preprocessing and alignment took {time.time() - start_preprocess:.2f}s")
            if fmri_aligned.shape[0] == 0 or video_aligned.shape[0] == 0:
                logger.warning(f"Alignment resulted in empty data for {subject_id}/{stimulus_lower}. Skipping.")
                continue

            save_data((fmri_aligned, video_aligned), aligned_cache_file) # Save to subject specific cache
            subject_fmri_aligned.append(fmri_aligned)
            subject_video_aligned.append(video_aligned)

        except Exception as e:
            logger.error(f"Error during preprocess/align for {subject_id}/{stimulus_lower}: {e}", exc_info=True)
            continue
        # --- End Stimulus Loop ---

    # --- Concatenate data for *this* subject ---
    if not subject_fmri_aligned:
        logger.warning(f"No data successfully processed for subject {subject_id}.")
        return None, None # Return None if no data for this subject

    final_fmri = np.concatenate(subject_fmri_aligned, axis=0)
    final_video = np.concatenate(subject_video_aligned, axis=0)
    logger.info(f"--- Final Concatenated Data Shapes for Subject {subject_id} ---")
    logger.info(f"Subject fMRI data: {final_fmri.shape}")
    logger.info(f"Subject Video embeddings: {final_video.shape}")
    return final_fmri, final_video

# --- Simplified Training Loop (No Validation) ---
# (You should place this in train.py or keep it here)
def run_training_no_val(model, train_loader, device, task='encoding', epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    """ Simplified training loop without validation. """
    model_type = "LSTM" if config.USE_TEMPORAL_MODELS else "MLP"
    logger.info(f"Starting {task} model training ({model_type}, no validation) for {epochs} epochs on {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_epoch_func = train_epoch # Assumes train_epoch is correctly defined (MLP version)

    for epoch in range(epochs):
        # Pass use_temporal flag if train_epoch still expects it, otherwise remove
        train_loss = train_epoch_func(model, train_loader, criterion, optimizer, device, task)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        # No validation step, no saving best model based on validation

    logger.info(f"Training complete after {epochs} epochs. Using model from last epoch.")
    return model # Return model from last epoch

# === Main Execution ===
def main():
    start_time_main = time.time()
    logger.info("--- Starting fMRI-Video Alignment Pipeline ---")

    # --- Determine and Set Video Encoder Configuration ---
    chosen_encoder = config.VIDEO_ENCODER_NAME
    if chosen_encoder not in config.VIDEO_MODEL_IDENTIFIERS:
        logger.error(f"Invalid VIDEO_ENCODER_NAME '{chosen_encoder}'. Choose from {list(config.VIDEO_MODEL_IDENTIFIERS.keys())}.")
        return
    config.VIDEO_EMBEDDING_MODEL = config.VIDEO_MODEL_IDENTIFIERS[chosen_encoder]
    config.DEC_OUTPUT_DIM = config.VIDEO_MODEL_OUTPUT_DIMS[chosen_encoder]
    logger.info(f"Selected Video Encoder: {chosen_encoder} (ID: {config.VIDEO_EMBEDDING_MODEL}, Dim: {config.DEC_OUTPUT_DIM})")

    # --- Log Core Config ---
    logger.info(f"Train Subject(s): {config.TRAIN_SUBJECT_IDS}, Test Subject: {config.TEST_SUBJECT_ID}")
    logger.info(f"Device: {config.DEVICE}, fMRI Variant: {config.FMRI_VARIANT}")
    logger.info(f"Apply PCA: {config.APPLY_PCA}, PCA Components: {config.PCA_N_COMPONENTS if config.APPLY_PCA else 'N/A'}")
    logger.info(f"Using Temporal Models: {config.USE_TEMPORAL_MODELS}") # Should be False for MLP

    # --- Determine TR ---
    # Use first training subject for TR lookup (assuming consistency)
    tr_lookup_subj = config.TRAIN_SUBJECT_IDS[0]
    try:
        reference_stimulus = config.STIMULI_NAMES[0]
        # fmri_tr = get_tr_from_json(tr_lookup_subj, reference_stimulus)
        fmri_tr = 1
        logger.info(f"Determined TR using {tr_lookup_subj}: {fmri_tr:.4f}s")
        config.TR = fmri_tr
    except Exception as e:
        logger.error(f"Failed to determine TR using subject {tr_lookup_subj}: {e}. Exiting.", exc_info=True)
        return

    # --- Initialize Video Extractor ---
    try:
        video_extractor = VideoFeatureExtractor(
            model_identifier=config.VIDEO_EMBEDDING_MODEL,
            device=config.DEVICE
        )
    except Exception as e:
        logger.error(f"Failed to initialize VideoFeatureExtractor in main: {e}", exc_info=True)
        return

    # --- Process Training Data ---
    all_train_fmri_raw = []
    all_train_video = []
    for train_subj_id in config.TRAIN_SUBJECT_IDS:
        fmri_subj_raw, video_subj = process_subject_data(train_subj_id, fmri_tr, video_extractor, chosen_encoder)
        if fmri_subj_raw is not None and video_subj is not None:
            all_train_fmri_raw.append(fmri_subj_raw)
            all_train_video.append(video_subj)
        else:
            logger.warning(f"Could not process data for training subject: {train_subj_id}")


    if not all_train_fmri_raw:
        logger.error("No training data could be processed. Exiting.")
        return

    final_train_fmri_raw = np.concatenate(all_train_fmri_raw, axis=0)
    final_train_video = np.concatenate(all_train_video, axis=0)
    logger.info(f"--- Combined Training Data Shapes (Before PCA) ---")
    logger.info(f"Train fMRI (Raw/SRM): {final_train_fmri_raw.shape}")
    logger.info(f"Train Video ({chosen_encoder}): {final_train_video.shape}")

    # --- Apply PCA (Fit on Training Data Only) ---
    scaler = None
    pca = None
    final_train_fmri = final_train_fmri_raw # Assign initially

    if config.APPLY_PCA:
        logger.info(f"Applying PCA (Fitting on Training Data). Target components: {config.PCA_N_COMPONENTS}")
        if final_train_fmri_raw.shape[1] <= config.PCA_N_COMPONENTS:
            logger.warning(f"Train features ({final_train_fmri_raw.shape[1]}) <= target PCA components. Skipping PCA fitting.")
        else:
            try:
                scaler = StandardScaler()
                fmri_train_scaled = scaler.fit_transform(final_train_fmri_raw)
                pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=42)
                start_pca_fit = time.time()
                final_train_fmri = pca.fit_transform(fmri_train_scaled) # Overwrite with PCA data
                logger.info(f"PCA Fitting/Transform took {time.time() - start_pca_fit:.2f}s.")
                logger.info(f"Explained variance on train data: {np.sum(pca.explained_variance_ratio_):.4f}")
                # TODO: Save PCA/Scaler models
            except Exception as e:
                 logger.error(f"Error during PCA fitting on training data: {e}. Proceeding without PCA.", exc_info=True)
                 config.APPLY_PCA = False # Disable PCA if fitting failed
                 final_train_fmri = final_train_fmri_raw # Revert to raw

    logger.info(f"--- Final Training Data Shapes (After PCA if Applied) ---")
    logger.info(f"Train fMRI data: {final_train_fmri.shape}")
    logger.info(f"Train Video embeddings: {final_train_video.shape}")

    # --- Prepare Training DataLoader ---
    try:
        train_dataset = TensorDataset(torch.from_numpy(final_train_fmri).float(),
                                      torch.from_numpy(final_train_video).float())
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        logger.info(f"Train DataLoader ready (Size: {len(train_dataset)} samples).")
    except Exception as e:
        logger.error(f"Failed to create training DataLoader: {e}", exc_info=True)
        return


    # --- Process Test Data ---
    final_test_fmri = None
    video_test = None
    if config.TEST_SUBJECT_ID:
        logger.info(f"--- Processing Test Data for Subject: {config.TEST_SUBJECT_ID} ---")
        fmri_test_raw, video_test = process_subject_data(config.TEST_SUBJECT_ID, fmri_tr, video_extractor, chosen_encoder)

        if fmri_test_raw is None or video_test is None:
             logger.error(f"No test data could be processed for subject {config.TEST_SUBJECT_ID}. Cannot evaluate.")
             # Decide whether to proceed with training only or exit
             return
        else:
            # --- Apply PCA Transformation (Using Scaler/PCA fitted on Train Data) ---
            if config.APPLY_PCA and scaler is not None and pca is not None:
                logger.info("Applying PCA transformation to Test fMRI data...")
                try:
                    if fmri_test_raw.shape[1] != scaler.n_features_in_:
                        logger.error(f"Test data feature count ({fmri_test_raw.shape[1]}) doesn't match scaler ({scaler.n_features_in_}). Cannot apply PCA.")
                        # Fallback? Exit? For now, try without PCA
                        final_test_fmri = fmri_test_raw
                    else:
                        start_pca_transform = time.time()
                        fmri_test_scaled = scaler.transform(fmri_test_raw)
                        final_test_fmri = pca.transform(fmri_test_scaled)
                        logger.info(f"PCA Transform took {time.time() - start_pca_transform:.2f}s.")
                except Exception as e:
                    logger.error(f"Error applying PCA transform to test data: {e}. Using raw test data.", exc_info=True)
                    final_test_fmri = fmri_test_raw
            else:
                 final_test_fmri = fmri_test_raw # Use raw test data if PCA wasn't applied/fitted

            logger.info(f"--- Final Test Data Shapes (After PCA if Applied) ---")
            logger.info(f"Test fMRI data: {final_test_fmri.shape}")
            logger.info(f"Test Video embeddings: {video_test.shape}")

            # --- Prepare Test DataLoader ---
            try:
                test_dataset = TensorDataset(torch.from_numpy(final_test_fmri).float(),
                                             torch.from_numpy(video_test).float())
                test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
                logger.info(f"Test DataLoader ready (Size: {len(test_dataset)} samples).")
            except Exception as e:
                 logger.error(f"Failed to create test DataLoader: {e}", exc_info=True)
                 test_loader = None # Mark as None if failed
    else:
         logger.info("No TEST_SUBJECT_ID specified. Skipping test phase.")
         test_loader = None

    # --- Update config dims (Based on final *training* data) ---
    config.ENC_OUTPUT_DIM = final_train_fmri.shape[1]
    config.DEC_INPUT_DIM = final_train_fmri.shape[1]
    # DEC_OUTPUT_DIM is already set based on video encoder choice

    # --- Model Initialization (MLP) ---
    if config.USE_TEMPORAL_MODELS: # Double check config consistency
        logger.error("Config mismatch: USE_TEMPORAL_MODELS should be False for MLP pipeline.")
        return
    logger.info("Initializing MLP Models...")
    encoding_model = EncodingModel(
        video_embed_dim=config.DEC_OUTPUT_DIM,
        fmri_dim=config.ENC_OUTPUT_DIM,
        hidden_dim=config.ENC_HIDDEN_DIM
    )
    decoding_model = DecodingModel(
        fmri_dim=config.DEC_INPUT_DIM,
        video_embed_dim=config.DEC_OUTPUT_DIM,
        hidden_dim=config.DEC_HIDDEN_DIM
    )

    # --- Encoding Model Training & Evaluation ---
    logger.info(f"\n--- Training Encoding Model (MLP, Video: {chosen_encoder}) on Subj {config.TRAIN_SUBJECT_IDS} ---")
    encoding_model = run_training_no_val( # Use modified training loop
        encoding_model, train_loader, device=config.DEVICE, task='encoding',
        epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE
    )

    if test_loader:
        logger.info(f"\n--- Evaluating Encoding Model on Subj {config.TEST_SUBJECT_ID} ---")
        enc_results, enc_targets, enc_predictions = evaluate_model(
             encoding_model, test_loader, nn.MSELoss(), device=config.DEVICE, task='encoding'
        )
        # --- Add encoder/subject info to output files ---
        enc_model_suffix = f"_mlp_{chosen_encoder}_train{''.join(config.TRAIN_SUBJECT_IDS)}_test{config.TEST_SUBJECT_ID}"
        enc_model_path = config.OUTPUT_DIR / f"encoding_model{enc_model_suffix}.pt"
        torch.save(encoding_model.state_dict(), enc_model_path)
        logger.info(f"Saved Encoding model to {enc_model_path}")

        enc_fig = plot_predictions(enc_targets, enc_predictions, n_samples=5, title=f"Encoding (MLP, {chosen_encoder}) Test Subj {config.TEST_SUBJECT_ID}")
        enc_plot_path = config.OUTPUT_DIR / f"encoding_predictions{enc_model_suffix}.png"
        enc_fig.savefig(enc_plot_path)
        plt.close(enc_fig)
        logger.info(f"Saved Encoding prediction plot to {enc_plot_path}")
    else:
         logger.info("Skipping encoding model evaluation as no test loader was created.")


    # --- Decoding Model Training & Evaluation ---
    logger.info(f"\n--- Training Decoding Model (MLP, Video: {chosen_encoder}) on Subj {config.TRAIN_SUBJECT_IDS} ---")
    if config.DEVICE == 'cuda' or 'cuda:0' or 'cuda:1':
        # Clear memory if possible
        del encoding_model
        if 'enc_targets' in locals(): del enc_targets
        if 'enc_predictions' in locals(): del enc_predictions
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before starting decoding model training.")

    decoding_model = run_training_no_val( # Use modified training loop
        decoding_model, train_loader, device=config.DEVICE, task='decoding',
        epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE
    )

    if test_loader:
        logger.info(f"\n--- Evaluating Decoding Model on Subj {config.TEST_SUBJECT_ID} ---")
        dec_results, dec_targets, dec_predictions = evaluate_model(
            decoding_model, test_loader, nn.MSELoss(), device=config.DEVICE, task='decoding'
        )
        dec_model_suffix = f"_mlp_{chosen_encoder}_train{''.join(config.TRAIN_SUBJECT_IDS)}_test{config.TEST_SUBJECT_ID}"
        dec_model_path = config.OUTPUT_DIR / f"decoding_model{dec_model_suffix}.pt"
        torch.save(decoding_model.state_dict(), dec_model_path)
        logger.info(f"Saved Decoding model to {dec_model_path}")

        dec_fig = plot_predictions(dec_targets, dec_predictions, n_samples=5, title=f"Decoding (MLP, {chosen_encoder}) Test Subj {config.TEST_SUBJECT_ID}")
        dec_plot_path = config.OUTPUT_DIR / f"decoding_predictions{dec_model_suffix}.png"
        dec_fig.savefig(dec_plot_path)
        plt.close(dec_fig)
        logger.info(f"Saved Decoding prediction plot to {dec_plot_path}")
    else:
         logger.info("Skipping decoding model evaluation as no test loader was created.")

    logger.info(f"\n--- Pipeline Finished --- Total Time: {(time.time() - start_time_main)/60:.2f} minutes ---")


if __name__ == "__main__":
    main()