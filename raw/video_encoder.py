# src/video_encoder.py
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification # Using VideoMAE as example
# Or use TimeSformer: from transformers import TimeSformerModel, VideoMAEImageProcessor (processor often shared/similar)
# from transformers import AutoProcessor, AutoModel # Generic approach
from transformers import AutoProcessor, AutoModel
import logging
from tqdm import tqdm
from typing import List
from config import VIDEO_EMBEDDING_MODEL, VIDEO_EMBEDDING_BATCH_SIZE, DEVICE, VIDEO_CHUNK_SIZE
import math
import config

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    def __init__(self, model_id: str, fps: float, device: str = "cuda", default_chunk_size: int = 16):
        self.model_id = model_id
        self.device = torch.device(device)
        self.fps = fps
        self.default_chunk_size = default_chunk_size

        logger.info(f"Loading model: {self.model_id} on {self.device}")
        if model_id == 'facebook/vit-mae-base':
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        else :
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

        self.num_frames_per_clip = self._determine_num_frames()
        self.stride = math.ceil(fps)  # one clip per second

    def _determine_num_frames(self) -> int:
        def get_nested_attr(obj, path):
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            return obj

        paths = [
            ('image_processor', 'num_frames'),
            ('config', 'num_frames'),
        ]
        for path in paths:
            val = get_nested_attr(self.processor, path)
            if val: return val
        val = get_nested_attr(self.model, ('config', 'num_frames'))
        if val: return val

        model_lower = self.model_id.lower()
        if "videomae" in model_lower or "vivit" in model_lower or "vitmae" in model_lower :
            return 16
        elif "timesformer" in model_lower or "xclip" in model_lower:
            return 8

        return self.default_chunk_size

    def _chunk_video(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        clips = []
        total = len(frames)
        for i in range(0, total - self.num_frames_per_clip + 1, self.stride):
            clip = frames[i:i + self.num_frames_per_clip]
            if len(clip) < self.num_frames_per_clip:
                clip += [clip[-1]] * (self.num_frames_per_clip - len(clip))
            clips.append(clip)
        return clips

    @torch.no_grad()
    def extract_features(self, video_frames: List[np.ndarray], batch_size: int = 4) -> np.ndarray:
        if not video_frames:
            return np.array([])

        expected_seconds = int(len(video_frames) / self.fps)
        clips = self._chunk_video(video_frames)
        if not clips:
            return np.array([])

        embeddings = []
        model_lower = self.model_id.lower()

        for i in tqdm(range(0, len(clips), batch_size), desc="Extracting"):
            batch_clips = clips[i:i + batch_size]

            try:
                if "vit-mae" in model_lower:
                    # Pick the middle frame from each clip
                    batch_frames = [clip[len(clip) // 2] for clip in batch_clips]
                    inputs = self.processor(images=batch_frames, return_tensors="pt")
                else:
                    # Assume video models can process full clips
                    inputs = self.processor(batch_clips, return_tensors="pt")

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                embed = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embed.cpu().numpy())

            except Exception as e:
                logger.error(f"Batch {i} failed: {e}", exc_info=True)
                continue

        if not embeddings:
            return np.array([])

        full_embeddings = np.concatenate(embeddings, axis=0)
        print('> full embedding size : ', full_embeddings.shape)
        print('> expected seconds : ', expected_seconds)

        if full_embeddings.shape[0] > expected_seconds:
            return full_embeddings[:expected_seconds]
        elif full_embeddings.shape[0] < expected_seconds:
            last = full_embeddings[-1:]
            padding = np.repeat(last, expected_seconds - full_embeddings.shape[0], axis=0)
            return np.vstack([full_embeddings, padding])

        print("Shape of the embeddings is ", full_embeddings.shape)
        return full_embeddings
