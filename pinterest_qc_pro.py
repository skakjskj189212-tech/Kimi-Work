#!/usr/bin/env python3
"""
Pinterest Reference Curation QC — Production AI Edition
========================================================

An AI-enhanced, config-driven quality control pipeline for curating Pinterest
reference images against a manually defined archetype. Uses CLIP, InsightFace,
and ensemble scoring to filter, rank, and categorize images at production quality.

Usage:
------
    1. Install dependencies:
       pip install opencv-python pillow numpy pandas pyyaml imagehash tqdm torch torchvision
       pip install transformers timm insightface  # AI models (~2GB download)

    2. Curate 5-10 archetype images in ./archetype_anchors/

    3. Bulk download Pinterest images to ./pinterest_raw/

    4. Run:
       python pinterest_qc_pro.py --config qc_config.yaml --archetype ./archetype_anchors --input ./pinterest_raw --output ./output

Output:
-------
    output/
    ├── face_anchors/       # Top-N close-up face shots (ComfyUI ReferenceLatent)
    ├── hair_refs/          # Best hair references
    ├── outfit_refs/         # Best outfit references
    ├── pose_refs/           # Best pose references
    ├── full_body_refs/      # Best full-body proportion references
    ├── approved_all/        # All passing images
    ├── review/              # Borderline images for manual inspection
    ├── reject/              # Failed images
    ├── report.csv           # Full scoring spreadsheet
    ├── report.json          # Per-image rich metadata
    └── qc.log               # Detailed processing log

Architecture:
-------------
    - Config-driven (YAML): all thresholds, weights, model selection
    - Modular scorers: each AI/CV scorer is independent and degrades gracefully
    - Ensemble scoring: weighted combination of 6+ scorers into 0-100 score
    - Rich output: JSON per image + CSV summary + organized folders
    - Production-ready: logging, error handling, progress tracking, parallel loading
"""

import os
import sys
import shutil
import json
import logging
import argparse
import base64
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time

# =============================================================================
# CORE DEPENDENCIES (hard requirements)
# =============================================================================

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"CRITICAL: Missing core dependency: {e}")
    print("  pip install opencv-python pillow numpy")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False

try:
    import imagehash
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

# =============================================================================
# AI MODEL DEPENDENCIES (soft requirements — degrade gracefully)
# =============================================================================

CLIP_OK = False
CLIP_PROCESSOR = None
CLIP_MODEL = None

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_OK = True
except ImportError:
    torch = None
    CLIP_OK = False

INSIGHTFACE_OK = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except ImportError:
    INSIGHTFACE_OK = False

MEDIAPIPE_OK = False
try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

FACENET_OK = False
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_OK = True
except ImportError:
    FACENET_OK = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Loaded from qc_config.yaml. All fields have sensible defaults."""
    character_name: str = "nyra"
    character_description: str = ""
    
    archetype_match_pass: float = 0.55
    archetype_match_review: float = 0.30
    skin_tone_max_delta: float = 12.0
    skin_tone_review_delta: float = 18.0
    
    blur_pass: float = 100.0
    blur_review: float = 30.0
    clip_black_reject: float = 25.0
    clip_white_reject: float = 20.0
    clip_review: float = 5.0
    
    face_quality_pass: float = 0.6
    face_quality_review: float = 0.3
    aesthetic_pass: float = 5.5
    aesthetic_review: float = 4.0
    
    dup_reject_dist: int = 2
    dup_review_dist: int = 8
    clip_dup_threshold: float = 0.92
    
    shoulder_hip_min: float = 0.7
    shoulder_hip_max: float = 1.6
    head_body_min: float = 0.08
    head_body_max: float = 0.45
    
    closeup_min: float = 0.35
    portrait_min: float = 0.15
    halfbody_min: float = 0.08
    
    weight_archetype_match: float = 0.25
    weight_semantic_clip: float = 0.15
    weight_skin_tone: float = 0.10
    weight_blur: float = 0.10
    weight_exposure: float = 0.05
    weight_aesthetic: float = 0.15
    weight_face_quality: float = 0.10
    weight_pose_quality: float = 0.10
    weight_ai_brain: float = 0.35  # Primary weight when using agy CLI
    
    overall_pass: float = 60.0
    overall_review: float = 40.0
    
    top_n_face_anchors: int = 10
    top_n_hair: int = 15
    top_n_outfit: int = 15
    top_n_pose: int = 15
    top_n_full_body: int = 10
    
    clip_model: str = "openai/clip-vit-large-patch14"
    insightface_model: str = "buffalo_l"
    use_dwpose: bool = False
    
    # AI Brain settings (agy / Gemini / CLI)
    ai_brain_mode: str = "cli"  # "local", "gemini", "cli"
    cli_command: str = 'agy --dangerously-skip-permissions --print "Analyze the image at \'{image_path}\': {prompt}"'
    cli_timeout: int = 120
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_timeout: int = 30
    vlm_questions: List[str] = field(default_factory=list)
    
    log_level: str = "INFO"
    print_summary: bool = True
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        if not YAML_OK:
            print("WARNING: pyyaml not installed. Using default config.")
            return cls()
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            if not data:
                return cls()
            cfg = cls()
            # Map YAML keys to dataclass fields
            for section in ['character', 'archetype', 'quality', 'duplicates', 'pose', 'shot_type', 'ensemble', 'output', 'models', 'ai_brain', 'logging']:
                if section in data and isinstance(data[section], dict):
                    for key, val in data[section].items():
                        attr = key
                        if section == 'ensemble' and not key.startswith('weight_'):
                            attr = f'weight_{key}'
                        elif section == 'ai_brain':
                            if key == 'mode':
                                attr = 'ai_brain_mode'
                            elif key == 'cli_command':
                                attr = 'cli_command'
                            elif key == 'cli_timeout':
                                attr = 'cli_timeout'
                            elif key == 'gemini_api_key':
                                attr = 'gemini_api_key'
                            elif key == 'gemini_model':
                                attr = 'gemini_model'
                            elif key == 'gemini_timeout':
                                attr = 'gemini_timeout'
                            elif key == 'questions':
                                attr = 'vlm_questions'
                        elif section == 'quality':
                            if key == 'clip_black_reject':
                                attr = 'clip_black_reject'
                            elif key == 'clip_white_reject':
                                attr = 'clip_white_reject'
                            elif key == 'clip_review':
                                attr = 'clip_review'
                            elif key == 'face_quality_pass':
                                attr = 'face_quality_pass'
                            elif key == 'face_quality_review':
                                attr = 'face_quality_review'
                            elif key == 'aesthetic_pass':
                                attr = 'aesthetic_pass'
                            elif key == 'aesthetic_review':
                                attr = 'aesthetic_review'
                        elif section == 'output':
                            if key == 'overall_pass':
                                attr = 'overall_pass'
                            elif key == 'overall_review':
                                attr = 'overall_review'
                            elif key.startswith('top_n_'):
                                attr = key
                        elif section == 'models':
                            if key == 'clip_model':
                                attr = 'clip_model'
                            elif key == 'insightface_model':
                                attr = 'insightface_model'
                            elif key == 'use_dwpose':
                                attr = 'use_dwpose'
                        elif section == 'logging':
                            if key == 'level':
                                attr = 'log_level'
                            elif key == 'print_summary':
                                attr = 'print_summary'
                        if hasattr(cfg, attr):
                            setattr(cfg, attr, val)
            return cfg
        except Exception as e:
            print(f"WARNING: Failed to load config from {path}: {e}. Using defaults.")
            return cls()

# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """Loads and manages all AI models with lazy initialization and graceful degradation."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.device = self._resolve_device()
        self.logger = logging.getLogger("ModelManager")
        
        # CLIP
        self.clip_processor = None
        self.clip_model = None
        
        # InsightFace
        self.insightface_app = None
        
        # FaceNet (fallback)
        self.mtcnn = None
        self.resnet = None
        
        # MediaPipe
        self.pose_detector = None
        
        # State
        self.clip_archetype_embedding = None
        self.facenet_archetype_embeddings = None
        self.archetype_skin_lab = None
        
        self._init_clip()
        self._init_insightface()
        self._init_facenet()
        self._init_mediapipe()
    
    def _resolve_device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _init_clip(self):
        if not CLIP_OK:
            self.logger.warning("CLIP not available. Install: pip install transformers")
            return
        try:
            self.logger.info(f"Loading CLIP model: {self.cfg.clip_model}...")
            self.clip_processor = CLIPProcessor.from_pretrained(self.cfg.clip_model)
            self.clip_model = CLIPModel.from_pretrained(self.cfg.clip_model).to(self.device).eval()
            self.logger.info("CLIP loaded successfully.")
        except Exception as e:
            self.logger.error(f"CLIP load failed: {e}")
            self.clip_processor = None
            self.clip_model = None
    
    def _init_insightface(self):
        if not INSIGHTFACE_OK:
            self.logger.warning("InsightFace not available. Install: pip install insightface")
            return
        try:
            self.logger.info(f"Loading InsightFace: {self.cfg.insightface_model}...")
            self.insightface_app = FaceAnalysis(
                name=self.cfg.insightface_model,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace loaded successfully.")
        except Exception as e:
            self.logger.error(f"InsightFace load failed: {e}")
            self.insightface_app = None
    
    def _init_facenet(self):
        if not FACENET_OK or self.insightface_app is not None:
            # Skip FaceNet if InsightFace is working (primary face model)
            if self.insightface_app is not None:
                self.logger.info("FaceNet skipped (InsightFace is primary).")
            return
        try:
            self.logger.info("Loading FaceNet as fallback...")
            self.mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                keep_all=True, device=self.device, post_process=True
            )
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.logger.info("FaceNet loaded successfully.")
        except Exception as e:
            self.logger.error(f"FaceNet load failed: {e}")
            self.mtcnn = None
            self.resnet = None
    
    def _init_mediapipe(self):
        if not MEDIAPIPE_OK:
            self.logger.warning("MediaPipe not available. Install: pip install mediapipe")
            return
        try:
            self.logger.info("Loading MediaPipe Pose...")
            self.pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True, model_complexity=1,
                min_detection_confidence=0.5
            )
            self.logger.info("MediaPipe loaded successfully.")
        except Exception as e:
            self.logger.error(f"MediaPipe load failed: {e}")
            self.pose_detector = None
    
    def compute_clip_image_embedding(self, pil_img: Image.Image) -> Optional[np.ndarray]:
        """Compute CLIP image embedding. Returns normalized numpy array."""
        if self.clip_model is None or self.clip_processor is None:
            return None
        try:
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            emb = image_features.cpu().numpy().flatten()
            emb = emb / np.linalg.norm(emb)
            return emb
        except Exception as e:
            self.logger.debug(f"CLIP image embedding failed: {e}")
            return None
    
    def compute_clip_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute CLIP text embedding. Returns normalized numpy array."""
        if self.clip_model is None or self.clip_processor is None:
            return None
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            emb = text_features.cpu().numpy().flatten()
            emb = emb / np.linalg.norm(emb)
            return emb
        except Exception as e:
            self.logger.debug(f"CLIP text embedding failed: {e}")
            return None
    
    def is_ready(self, model_name: str) -> bool:
        checks = {
            "clip": self.clip_model is not None,
            "insightface": self.insightface_app is not None,
            "facenet": self.mtcnn is not None and self.resnet is not None,
            "mediapipe": self.pose_detector is not None,
        }
        return checks.get(model_name.lower(), False)

# =============================================================================
# SCORER CLASSES (each returns: score, status, note)
# =============================================================================

class ScorerResult:
    def __init__(self, score: float, status: str, note: str, raw: Any = None):
        self.score = score          # Normalized 0-1, or -1 if unavailable
        self.status = status        # PASS, REVIEW, REJECT
        self.note = note            # Human-readable diagnostic
        self.raw = raw              # Optional raw data

class BaseScorer:
    def __init__(self, config: Config, models: ModelManager):
        self.cfg = config
        self.models = models
        self.logger = logging.getLogger(self.__class__.__name__)

# --- 1. InsightFace Scorer (primary face recognition + quality) ---

class InsightFaceScorer(BaseScorer):
    def __init__(self, config: Config, models: ModelManager, archetype_dir: Path):
        super().__init__(config, models)
        self.archetype_dir = archetype_dir
        self.archetype_embeddings = []
        self._load_archetype_embeddings()
    
    def _load_archetype_embeddings(self):
        if not self.models.is_ready("insightface"):
            return
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = [p for p in self.archetype_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_ext]
        self.logger.info(f"Loading {len(paths)} archetype images for InsightFace...")
        for p in paths:
            try:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                faces = self.models.insightface_app.get(img)
                if faces and len(faces) > 0:
                    # Use largest face by bbox area
                    largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                    self.archetype_embeddings.append(largest.embedding)
                    self.logger.debug(f"  Archetype face loaded: {p.name} (det_score={largest.det_score:.3f})")
            except Exception as e:
                self.logger.debug(f"  Failed archetype {p.name}: {e}")
        self.logger.info(f"Loaded {len(self.archetype_embeddings)} archetype face embeddings.")
    
    def score(self, cv_img: np.ndarray) -> ScorerResult:
        if not self.models.is_ready("insightface"):
            return ScorerResult(-1, "PASS", "insightface_unavailable")
        if not self.archetype_embeddings:
            return ScorerResult(-1, "PASS", "archetype_embeddings_missing")
        
        try:
            faces = self.models.insightface_app.get(cv_img)
            if not faces or len(faces) == 0:
                return ScorerResult(0.0, "REJECT", "no_face_detected")
            
            # Primary: largest face
            primary = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            
            # Quality checks
            det_score = getattr(primary, 'det_score', 0.5)
            face_quality = getattr(primary, 'face_quality', 0.5)
            
            # Compare to archetype embeddings (cosine similarity)
            candidate_emb = primary.embedding
            similarities = []
            for arche_emb in self.archetype_embeddings:
                sim = np.dot(candidate_emb, arche_emb) / (np.linalg.norm(candidate_emb) * np.linalg.norm(arche_emb) + 1e-8)
                similarities.append(sim)
            best_sim = max(similarities) if similarities else 0.0
            
            # Combined score: 70% similarity + 30% quality
            combined = 0.7 * best_sim + 0.3 * face_quality
            
            if best_sim >= self.cfg.archetype_match_pass and face_quality >= self.cfg.face_quality_pass:
                return ScorerResult(combined, "PASS", f"sim={best_sim:.3f} qual={face_quality:.3f} det={det_score:.3f}")
            if best_sim >= self.cfg.archetype_match_review:
                return ScorerResult(combined, "REVIEW", f"sim={best_sim:.3f} qual={face_quality:.3f}")
            return ScorerResult(combined, "REJECT", f"sim={best_sim:.3f} (too low)")
            
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"insightface_error: {e}")

# --- 2. CLIP Semantic Scorer (image-text alignment) ---

class CLIPSemanticScorer(BaseScorer):
    def __init__(self, config: Config, models: ModelManager):
        super().__init__(config, models)
        self.text_embedding = None
        if self.models.is_ready("clip"):
            self.text_embedding = self.models.compute_clip_text_embedding(config.character_description)
            if self.text_embedding is not None:
                self.logger.info(f"CLIP text embedding computed for: '{config.character_description[:60]}...'")
    
    def score(self, pil_img: Image.Image) -> ScorerResult:
        if not self.models.is_ready("clip"):
            return ScorerResult(-1, "PASS", "clip_unavailable")
        if self.text_embedding is None:
            return ScorerResult(-1, "PASS", "clip_text_embedding_missing")
        
        try:
            img_emb = self.models.compute_clip_image_embedding(pil_img)
            if img_emb is None:
                return ScorerResult(-1, "REVIEW", "clip_image_embedding_failed")
            
            similarity = float(np.dot(img_emb, self.text_embedding))
            # Clip similarity is typically 0.2-0.4 for good matches
            # Normalize to 0-1: threshold around 0.25 as baseline
            normalized = max(0.0, min(1.0, (similarity - 0.15) / 0.25))
            
            if similarity >= 0.25:
                return ScorerResult(normalized, "PASS", f"clip_sim={similarity:.3f}")
            if similarity >= 0.20:
                return ScorerResult(normalized, "REVIEW", f"clip_sim={similarity:.3f}")
            return ScorerResult(normalized, "REJECT", f"clip_sim={similarity:.3f} (wrong subject)")
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"clip_error: {e}")

# --- 3. CLIP Aesthetic Scorer ---

class CLIPAestheticScorer(BaseScorer):
    """Uses CLIP to compare image against positive and negative aesthetic descriptors."""
    
    def __init__(self, config: Config, models: ModelManager):
        super().__init__(config, models)
        self.positive_emb = None
        self.negative_emb = None
        if self.models.is_ready("clip"):
            pos_text = "high quality professional photograph, sharp focus, beautiful lighting, photorealistic, masterpiece, best quality"
            neg_text = "low quality, blurry, jpeg artifacts, bad anatomy, oversaturated, plastic skin, amateur photo"
            self.positive_emb = self.models.compute_clip_text_embedding(pos_text)
            self.negative_emb = self.models.compute_clip_text_embedding(neg_text)
    
    def score(self, pil_img: Image.Image) -> ScorerResult:
        if not self.models.is_ready("clip") or self.positive_emb is None:
            return ScorerResult(-1, "PASS", "clip_aesthetic_unavailable")
        
        try:
            img_emb = self.models.compute_clip_image_embedding(pil_img)
            if img_emb is None:
                return ScorerResult(-1, "REVIEW", "clip_image_embedding_failed")
            
            pos_sim = float(np.dot(img_emb, self.positive_emb))
            neg_sim = float(np.dot(img_emb, self.negative_emb))
            
            # Aesthetic score: positive - negative, scaled to 0-10
            raw_score = (pos_sim - neg_sim + 0.2) * 15.0
            score_10 = max(0.0, min(10.0, raw_score))
            normalized = score_10 / 10.0
            
            if score_10 >= self.cfg.aesthetic_pass:
                return ScorerResult(normalized, "PASS", f"aesthetic={score_10:.1f}/10")
            if score_10 >= self.cfg.aesthetic_review:
                return ScorerResult(normalized, "REVIEW", f"aesthetic={score_10:.1f}/10")
            return ScorerResult(normalized, "REJECT", f"aesthetic={score_10:.1f}/10 (low quality)")
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"aesthetic_error: {e}")

# --- 4. Blur Scorer ---

class BlurScorer(BaseScorer):
    def score(self, cv_img: np.ndarray) -> ScorerResult:
        if cv_img is None:
            return ScorerResult(0.0, "REVIEW", "image_missing")
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize: 30 = 0, 200 = 1.0
            normalized = max(0.0, min(1.0, (lap_var - 30) / 170))
            
            if lap_var >= self.cfg.blur_pass:
                return ScorerResult(normalized, "PASS", f"lapvar={lap_var:.1f}")
            if lap_var >= self.cfg.blur_review:
                return ScorerResult(normalized, "REVIEW", f"lapvar={lap_var:.1f}")
            return ScorerResult(normalized, "REJECT", f"lapvar={lap_var:.1f} (too blurry)")
        except Exception as e:
            return ScorerResult(0.0, "REVIEW", f"blur_error: {e}")

# --- 5. Exposure Scorer ---

class ExposureScorer(BaseScorer):
    def score(self, cv_img: np.ndarray) -> ScorerResult:
        if cv_img is None:
            return ScorerResult(0.0, "REVIEW", "image_missing")
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            total = gray.size
            black = float(np.sum(gray < 10) / total * 100)
            white = float(np.sum(gray > 245) / total * 100)
            
            # Exposure score: penalize clipping, reward balanced histogram
            # Ideal: black < 5%, white < 5%
            black_penalty = max(0, black - 5) / 20  # 0-1 penalty
            white_penalty = max(0, white - 5) / 20
            normalized = max(0.0, 1.0 - (black_penalty + white_penalty) / 2)
            
            if black > self.cfg.clip_black_reject or white > self.cfg.clip_white_reject:
                return ScorerResult(normalized, "REJECT", f"black={black:.1f}% white={white:.1f}%")
            if black > self.cfg.clip_review or white > self.cfg.clip_review:
                return ScorerResult(normalized, "REVIEW", f"black={black:.1f}% white={white:.1f}%")
            return ScorerResult(normalized, "PASS", f"black={black:.1f}% white={white:.1f}%")
        except Exception as e:
            return ScorerResult(0.0, "REVIEW", f"exposure_error: {e}")

# --- 6. Skin Tone Scorer ---

class SkinToneScorer(BaseScorer):
    def __init__(self, config: Config, models: ModelManager, archetype_dir: Path):
        super().__init__(config, models)
        self.archetype_skin_lab = self._compute_archetype_skin(archetype_dir)
        if self.archetype_skin_lab is not None:
            self.logger.info(f"Archetype skin tone LAB: {self.archetype_skin_lab}")
    
    def _compute_archetype_skin(self, archetype_dir: Path) -> Optional[np.ndarray]:
        labs = []
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = [p for p in archetype_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_ext]
        for p in paths:
            try:
                cv_img = cv2.imread(str(p))
                if cv_img is None:
                    continue
                lab = self._extract_skin(cv_img, use_center=True)
                if lab is not None:
                    labs.append(lab)
            except Exception:
                continue
        if labs:
            return np.mean(labs, axis=0)
        return None
    
    def _extract_skin(self, cv_img: np.ndarray, use_center: bool = False) -> Optional[np.ndarray]:
        try:
            if use_center:
                h, w = cv_img.shape[:2]
                cy, cx = h // 2, w // 2
                crop = cv_img[max(0, cy - h//3):min(h, cy + h//3), max(0, cx - w//3):min(w, cx + w//3)]
            else:
                crop = cv_img
            
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            mask = (
                (ycrcb[:, :, 1] >= 135) & (ycrcb[:, :, 1] <= 180) &
                (ycrcb[:, :, 2] >= 85) & (ycrcb[:, :, 2] <= 135)
            )
            skin_pixels = crop[mask]
            if len(skin_pixels) < 50:
                return None
            lab_pixels = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            return np.mean(lab_pixels, axis=0).astype(np.float32)
        except Exception:
            return None
    
    def score(self, cv_img: np.ndarray) -> ScorerResult:
        if self.archetype_skin_lab is None:
            return ScorerResult(-1, "PASS", "archetype_skin_unavailable")
        if cv_img is None:
            return ScorerResult(-1, "REVIEW", "image_missing")
        try:
            candidate_lab = self._extract_skin(cv_img, use_center=False)
            if candidate_lab is None:
                return ScorerResult(-1, "REVIEW", "no_skin_detected")
            delta = float(np.linalg.norm(self.archetype_skin_lab - candidate_lab))
            # Normalize: 0 delta = 1.0, 25 delta = 0.0
            normalized = max(0.0, 1.0 - delta / 25.0)
            
            if delta <= self.cfg.skin_tone_max_delta:
                return ScorerResult(normalized, "PASS", f"skin_delta={delta:.1f}")
            if delta <= self.cfg.skin_tone_review_delta:
                return ScorerResult(normalized, "REVIEW", f"skin_delta={delta:.1f}")
            return ScorerResult(normalized, "REJECT", f"skin_delta={delta:.1f} (wrong tone)")
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"skin_error: {e}")

# --- 7. Pose Scorer ---

class PoseScorer(BaseScorer):
    def score(self, cv_img: np.ndarray) -> ScorerResult:
        if not self.models.is_ready("mediapipe"):
            return ScorerResult(-1, "PASS", "mediapipe_unavailable")
        if cv_img is None:
            return ScorerResult(-1, "REVIEW", "image_missing")
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.models.pose_detector.process(rgb)
            if not results.pose_landmarks:
                return ScorerResult(-1, "REVIEW", "no_pose_detected")
            
            lm = results.pose_landmarks.landmark
            def get(name):
                return lm[mp.solutions.pose.PoseLandmark[name].value]
            
            key_points = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
            if any(get(k).visibility < 0.5 for k in key_points):
                return ScorerResult(-1, "PASS", "partial_pose")
            
            shoulder_w = abs(get("LEFT_SHOULDER").x - get("RIGHT_SHOULDER").x)
            hip_w = abs(get("LEFT_HIP").x - get("RIGHT_HIP").x)
            sh_ratio = shoulder_w / hip_w if hip_w > 0 else 0
            
            head_h = abs(get("NOSE").y - (get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2)
            body_h = abs((get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2 -
                       (get("LEFT_HIP").y + get("RIGHT_HIP").y) / 2)
            hb_ratio = head_h / body_h if body_h > 0 else 0
            
            # Score based on how close ratios are to ideal ranges
            sh_center = (self.cfg.shoulder_hip_min + self.cfg.shoulder_hip_max) / 2
            hb_center = (self.cfg.head_body_min + self.cfg.head_body_max) / 2
            sh_dev = abs(sh_ratio - sh_center) / (self.cfg.shoulder_hip_max - self.cfg.shoulder_hip_min)
            hb_dev = abs(hb_ratio - hb_center) / (self.cfg.head_body_max - self.cfg.head_body_min)
            normalized = max(0.0, 1.0 - (sh_dev + hb_dev) / 2)
            
            notes = []
            status = "PASS"
            if not (self.cfg.shoulder_hip_min <= sh_ratio <= self.cfg.shoulder_hip_max):
                notes.append(f"sh_ratio={sh_ratio:.2f}")
                status = "REVIEW"
            if not (self.cfg.head_body_min <= hb_ratio <= self.cfg.head_body_max):
                notes.append(f"hb_ratio={hb_ratio:.2f}")
                status = "REVIEW"
            
            return ScorerResult(normalized, status, "; ".join(notes) if notes else "proportions_ok")
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"pose_error: {e}")

# --- 8. Duplicate Scorer ---

class DuplicateScorer(BaseScorer):
    def __init__(self, config: Config, models: ModelManager):
        super().__init__(config, models)
        self.seen_hashes: List[Tuple[Any, Any]] = []
        self.seen_clip_embeddings: List[np.ndarray] = []
    
    def score(self, pil_img: Image.Image) -> ScorerResult:
        if not IMAGEHASH_OK and not self.models.is_ready("clip"):
            return ScorerResult(-1, "PASS", "duplicate_tools_unavailable")
        
        is_dup = False
        status = "PASS"
        note = "unique"
        
        # ImageHash check
        if IMAGEHASH_OK and pil_img is not None:
            try:
                phash = imagehash.phash(pil_img)
                dhash = imagehash.dhash(pil_img)
                for prev_phash, prev_dhash in self.seen_hashes:
                    pdist = phash - prev_phash
                    ddist = dhash - prev_dhash
                    if pdist <= self.cfg.dup_reject_dist and ddist <= self.cfg.dup_reject_dist:
                        return ScorerResult(0.0, "REJECT", f"exact_dup_p:{pdist}_d:{ddist}")
                    if pdist <= self.cfg.dup_review_dist and ddist <= self.cfg.dup_review_dist:
                        is_dup = True
                        status = "REVIEW"
                        note = f"near_dup_phash:{pdist}"
                self.seen_hashes.append((phash, dhash))
            except Exception:
                pass
        
        # CLIP embedding check (catches semantic duplicates even after cropping)
        if self.models.is_ready("clip") and pil_img is not None:
            try:
                emb = self.models.compute_clip_image_embedding(pil_img)
                if emb is not None:
                    for prev_emb in self.seen_clip_embeddings:
                        sim = float(np.dot(emb, prev_emb))
                        if sim >= self.cfg.clip_dup_threshold:
                            return ScorerResult(0.0, "REJECT", f"semantic_dup_clip:{sim:.3f}")
                        elif sim >= 0.85:
                            is_dup = True
                            status = "REVIEW"
                            note = f"semantic_near_dup_clip:{sim:.3f}"
                    self.seen_clip_embeddings.append(emb)
            except Exception:
                pass
        
        return ScorerResult(1.0 if not is_dup else 0.5, status, note)

# --- 9. CLI AI Brain (Google Antigravity / agy) ---

class CLIAnalyzer(BaseScorer):
    """
    Uses a local CLI tool as the primary AI brain.
    Default configured for Google Antigravity (agy) with Gemini 3.1 Pro.
    Falls back to text parsing if JSON extraction fails.
    """
    
    def __init__(self, config: Config, models: ModelManager):
        super().__init__(config, models)
        self.command_template = getattr(config, 'cli_command', '')
        if not self.command_template or self.command_template.strip() == "":
            self.command_template = 'agy --dangerously-skip-permissions --print "Analyze the image at \'{image_path}\': {prompt}"'
        self.timeout = getattr(config, 'cli_timeout', 120)
        self.enabled = config.ai_brain_mode == "cli" and bool(self.command_template)
        self.prompt_template = """You are a visual reference curator for an AI character dataset. Analyze the image and answer 5 questions about whether it matches this character archetype.

Character: {character_description}

Questions (answer yes or no for each):
1. Does the image show a woman with light brown to dark blonde hair?
2. Does the image show a woman with hazel-green or green eyes?
3. Does the image show a woman with warm beige or light olive skin tone?
4. Does the image show a woman with an oval or heart-shaped face and high cheekbones?
5. Is this a high-quality, photorealistic portrait photograph (not illustration or AI art)?

After answering, provide ONLY a JSON summary (no extra text, no markdown code blocks):
{{"face_match": 0.0-1.0, "hair_match": 0.0-1.0, "eye_match": 0.0-1.0, "skin_match": 0.0-1.0, "quality": 0.0-1.0, "is_photorealistic": true/false, "explanation": "brief reason"}}"""
    
    def score(self, pil_img: Image.Image) -> ScorerResult:
        if not self.enabled:
            return ScorerResult(-1, "PASS", "cli_not_configured")
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            pil_img.save(temp_path, format="JPEG", quality=90)
        
        try:
            prompt = self.prompt_template.format(
                character_description=getattr(self.cfg, 'character_description', 
                    'attractive young woman, light brown to dark blonde hair, hazel-green eyes, warm beige skin, oval face with high cheekbones')
            )
            
            # Substitute placeholders in command template
            cmd = self.command_template.replace('{prompt}', prompt)
            cmd = cmd.replace('{image_path}', temp_path)
            
            self.logger.info(f"Running AI brain (agy): {cmd[:100]}...")
            start_time = time.time()
            
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.timeout
            )
            
            elapsed = time.time() - start_time
            stdout = result.stdout
            stderr = result.stderr
            
            self.logger.debug(f"agy stdout ({len(stdout)} chars): {stdout[:500]}")
            self.logger.debug(f"agy stderr ({len(stderr)} chars): {stderr[:500]}")
            
            # Try to extract JSON from output
            json_data = self._extract_json(stdout)
            if json_data:
                return self._parse_json_scores(json_data, elapsed)
            
            # Fallback: parse text for yes/no counts
            return self._parse_text_output(stdout, stderr, elapsed)
            
        except subprocess.TimeoutExpired:
            return ScorerResult(-1, "REVIEW", f"agy_timeout_after_{self.timeout}s")
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"agy_error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from potentially messy CLI output."""
        import re
        # Try JSON inside markdown code blocks first
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # Try to find any JSON object in the text
        try:
            # Find the outermost { ... } pair
            brace_match = re.search(r'\{[\s\S]*\}', text)
            if brace_match:
                return json.loads(brace_match.group(0))
        except:
            pass
        
        return None
    
    def _parse_json_scores(self, data: dict, elapsed: float) -> ScorerResult:
        """Parse structured JSON from agy into a ScorerResult."""
        try:
            face_match = float(data.get('face_match', data.get('face', 0.5)))
            hair_match = float(data.get('hair_match', data.get('hair', 0.5)))
            eye_match = float(data.get('eye_match', data.get('eye', 0.5)))
            skin_match = float(data.get('skin_match', data.get('skin', 0.5)))
            quality = float(data.get('quality', data.get('overall_quality', 0.5)))
            is_photo = data.get('is_photorealistic', data.get('photorealistic', True))
            explanation = data.get('explanation', data.get('reason', 'agy analysis'))
            
            match_scores = [face_match, hair_match, eye_match, skin_match]
            avg_match = sum(match_scores) / len(match_scores)
            composite = 0.7 * avg_match + 0.3 * quality
            if not is_photo:
                composite *= 0.5
            
            if composite >= 0.6:
                status = "PASS"
            elif composite >= 0.35:
                status = "REVIEW"
            else:
                status = "REJECT"
            
            return ScorerResult(
                composite, status,
                f"agy({elapsed:.1f}s): {explanation[:60]} | match={avg_match:.2f} qual={quality:.2f}"
            )
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"agy_json_parse_failed: {e}")
    
    def _parse_text_output(self, stdout: str, stderr: str, elapsed: float) -> ScorerResult:
        """Fallback parser when JSON extraction fails. Counts yes/no answers."""
        text = (stdout + stderr).lower()
        
        # Count explicit yes/no answers
        yes_count = text.count('yes')
        no_count = text.count('no')
        total = yes_count + no_count
        
        if total == 0:
            # No clear yes/no — try to find numeric scores
            import re
            scores = re.findall(r'(\d+\.?\d*)\s*/\s*10', text)
            if scores:
                avg_score = sum(float(s) for s in scores) / len(scores) / 10.0
                return ScorerResult(avg_score, "REVIEW", f"agy_text_score({elapsed:.1f}s): {avg_score:.2f}")
            return ScorerResult(-1, "REVIEW", f"agy_output_unparseable({elapsed:.1f}s)")
        
        score = yes_count / total
        if score >= 0.7:
            status = "PASS"
        elif score >= 0.4:
            status = "REVIEW"
        else:
            status = "REJECT"
        
        return ScorerResult(
            score, status,
            f"agy_text({elapsed:.1f}s): yes={yes_count} no={no_count}"
        )


class GeminiAnalyzer(CLIAnalyzer):
    """
    Uses the official Google GenAI SDK (google-genai) to call Gemini API for image analysis.
    """
    
    def __init__(self, config: Config, models: ModelManager):
        super().__init__(config, models)
        self.api_key = getattr(config, 'gemini_api_key', '') or os.environ.get('GEMINI_API_KEY', '')
        self.model_name = getattr(config, 'gemini_model', 'gemini-1.5-flash')
        self.timeout = getattr(config, 'gemini_timeout', 30)
        self.enabled = config.ai_brain_mode == "gemini" and bool(self.api_key)
        
    def score(self, pil_img: Image.Image) -> ScorerResult:
        if not self.enabled:
            return ScorerResult(-1, "PASS", "gemini_api_key_not_configured")
        if pil_img is None:
            return ScorerResult(-1, "REVIEW", "image_missing")
            
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            
            prompt = self.prompt_template.format(
                character_description=getattr(self.cfg, 'character_description', 
                    'attractive young woman, light brown to dark blonde hair, hazel-green eyes, warm beige skin, oval face with high cheekbones')
            )
            
            self.logger.info(f"Running AI brain (Gemini API) using {self.model_name}...")
            start_time = time.time()
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=[pil_img, prompt]
            )
            
            elapsed = time.time() - start_time
            text = response.text
            self.logger.debug(f"Gemini response: {text}")
            
            # Extract JSON from output
            json_data = self._extract_json(text)
            if json_data:
                return self._parse_json_scores(json_data, elapsed)
            
            # Fallback: parse text for yes/no counts
            return self._parse_text_output(text, "", elapsed)
            
        except Exception as e:
            return ScorerResult(-1, "REVIEW", f"gemini_error: {e}")


# =============================================================================
# SHOT TYPE CLASSIFIER
# =============================================================================

class ShotTypeClassifier:
    """Classifies images by shot type: closeup, portrait, half_body, full_body, environmental."""
    
    def __init__(self, config: Config, models: ModelManager):
        self.cfg = config
        self.models = models
    
    def classify(self, cv_img: np.ndarray) -> str:
        if cv_img is None:
            return "unknown"
        h, w = cv_img.shape[:2]
        
        # Try to detect face and measure its size
        face_ratio = self._get_face_ratio(cv_img)
        
        if face_ratio is not None:
            if face_ratio >= self.cfg.closeup_min:
                return "closeup"
            if face_ratio >= self.cfg.portrait_min:
                return "portrait"
            if face_ratio >= self.cfg.halfbody_min:
                return "half_body"
        
        # Check for full body
        if self._has_full_body(cv_img):
            return "full_body"
        
        return "environmental"
    
    def _get_face_ratio(self, cv_img: np.ndarray) -> Optional[float]:
        if self.models.is_ready("insightface"):
            try:
                faces = self.models.insightface_app.get(cv_img)
                if faces and len(faces) > 0:
                    largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                    x1, y1, x2, y2 = largest.bbox
                    h, w = cv_img.shape[:2]
                    face_area = (x2 - x1) * (y2 - y1)
                    return face_area / (w * h)
            except Exception:
                pass
        return None
    
    def _has_full_body(self, cv_img: np.ndarray) -> bool:
        if not self.models.is_ready("mediapipe"):
            return False
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.models.pose_detector.process(rgb)
            if not results.pose_landmarks:
                return False
            lm = results.pose_landmarks.landmark
            has_shoulders = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5
            has_hips = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].visibility > 0.5
            has_legs = (
                lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.5 or
                lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.5
            )
            return has_shoulders and has_hips and has_legs
        except Exception:
            return False

# =============================================================================
# ENSEMBLE SCORER & CATEGORY ASSIGNMENT
# =============================================================================

class EnsembleScorer:
    """Combines all scorer results into a single 0-100 production score."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.weights = {
            "ai_brain": config.weight_ai_brain,
            "archetype": config.weight_archetype_match,
            "semantic": config.weight_semantic_clip,
            "skin": config.weight_skin_tone,
            "blur": config.weight_blur,
            "exposure": config.weight_exposure,
            "aesthetic": config.weight_aesthetic,
            "face_quality": config.weight_face_quality,
            "pose": config.weight_pose_quality,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"WARNING: Ensemble weights sum to {total}, not 1.0. Normalizing.")
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def compute(self, scores: Dict[str, ScorerResult]) -> float:
        total = 0.0
        active_weights = 0.0
        
        for key, result in scores.items():
            weight = self.weights.get(key, 0)
            if weight > 0 and result.score >= 0:
                total += result.score * weight
                active_weights += weight
        
        if active_weights == 0:
            return 50.0  # neutral if no scorers active
        
        raw = total / active_weights
        # Any REJECT status immediately caps score below review threshold
        statuses = [r.status for r in scores.values()]
        if "REJECT" in statuses:
            raw = min(raw, self.cfg.overall_review / 100.0 - 0.05)
        
        return max(0.0, min(100.0, raw * 100))

class CategoryAssigner:
    """Assigns the best reference category for each image."""
    
    CATEGORIES = ["face_anchor", "hair", "outfit", "pose", "full_body", "approved", "review", "reject"]
    
    def assign(self, record: Dict[str, Any], config: Config) -> str:
        overall = record.get("overall_status", "REJECT")
        if overall == "REJECT":
            return "reject"
        
        shot = record.get("shot_type", "unknown")
        arche_status = record.get("insightface_status", record.get("archetype_status", "PASS"))
        semantic_status = record.get("semantic_clip_status", "PASS")
        
        good_match = (arche_status == "PASS" or semantic_status == "PASS")
        
        if not good_match and overall == "REVIEW":
            return "review"
        
        if shot == "closeup" and arche_status == "PASS":
            return "face_anchor"
        if shot == "portrait":
            if arche_status == "PASS":
                return "hair" if record.get("pose_status") == "PASS" else "face_anchor"
            return "hair"
        if shot == "half_body":
            if arche_status == "PASS" and record.get("pose_status") == "PASS":
                return "outfit"
            return "pose"
        if shot == "full_body":
            if record.get("pose_status") == "PASS":
                return "full_body"
            return "pose"
        if overall == "PASS":
            return "approved"
        return "review"

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class ProductionPipeline:
    def __init__(self, config: Config, archetype_dir: Path, input_dir: Path, output_dir: Path):
        self.cfg = config
        self.archetype_dir = archetype_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results: List[Dict[str, Any]] = []
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("ProductionPipeline")
        
        # Initialize models
        self.logger.info("Initializing AI models...")
        self.models = ModelManager(config)
        
        # Initialize scorers
        self.scorers = {
            "archetype": InsightFaceScorer(config, self.models, archetype_dir),
            "semantic": CLIPSemanticScorer(config, self.models),
            "aesthetic": CLIPAestheticScorer(config, self.models),
            "skin": SkinToneScorer(config, self.models, archetype_dir),
            "blur": BlurScorer(config, self.models),
            "exposure": ExposureScorer(config, self.models),
            "pose": PoseScorer(config, self.models),
        }
        
        # Integrate AI Brain scorer depending on mode
        if config.ai_brain_mode == "cli":
            self.scorers["ai_brain"] = CLIAnalyzer(config, self.models)
        elif config.ai_brain_mode == "gemini":
            self.scorers["ai_brain"] = GeminiAnalyzer(config, self.models)

        self.dup_scorer = DuplicateScorer(config, self.models)
        self.shot_classifier = ShotTypeClassifier(config, self.models)
        self.ensemble = EnsembleScorer(config)
        self.category_assigner = CategoryAssigner()
        
        self.logger.info("Pipeline ready.")
    
    def _setup_logging(self):
        level = getattr(logging, self.cfg.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "qc.log", mode="w"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _load_images(self, img_path: Path) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Load both PIL and OpenCV versions of an image."""
        pil_img = None
        cv_img = None
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.logger.warning(f"Pillow failed on {img_path.name}: {e}")
        try:
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise ValueError("cv2.imread returned None")
        except Exception as e:
            self.logger.warning(f"OpenCV failed on {img_path.name}: {e}")
        return pil_img, cv_img
    
    def process(self):
        # Create output folders
        for sub in ("face_anchors", "hair_refs", "outfit_refs", "pose_refs",
                    "full_body_refs", "approved_all", "review", "reject"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_paths = sorted([p for p in self.input_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_ext])
        if not image_paths:
            self.logger.error(f"No images found in {self.input_dir}")
            return
        
        self.logger.info(f"Processing {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths, desc="QC Processing"):
            pil_img, cv_img = self._load_images(img_path)
            if pil_img is None and cv_img is None:
                self.logger.error(f"Cannot load {img_path.name}. Skipping.")
                continue
            
            # Run all scorers
            scores = {}
            scores["archetype"] = self.scorers["archetype"].score(cv_img)
            scores["semantic"] = self.scorers["semantic"].score(pil_img)
            scores["aesthetic"] = self.scorers["aesthetic"].score(pil_img)
            scores["skin"] = self.scorers["skin"].score(cv_img)
            scores["blur"] = self.scorers["blur"].score(cv_img)
            scores["exposure"] = self.scorers["exposure"].score(cv_img)
            scores["pose"] = self.scorers["pose"].score(cv_img)
            
            if "ai_brain" in self.scorers:
                scores["ai_brain"] = self.scorers["ai_brain"].score(pil_img)
                
            dup_result = self.dup_scorer.score(pil_img)
            
            # Shot type
            shot_type = self.shot_classifier.classify(cv_img)
            
            # Compute ensemble score
            overall_score = self.ensemble.compute(scores)
            
            # Determine overall status
            statuses = [s.status for s in scores.values()] + [dup_result.status]
            if "REJECT" in statuses:
                overall_status = "REJECT"
            elif "REVIEW" in statuses or overall_score < self.cfg.overall_review:
                overall_status = "REVIEW"
            elif overall_score >= self.cfg.overall_pass:
                overall_status = "PASS"
            else:
                overall_status = "REVIEW"
            
            # Build record
            record = {
                "filename": img_path.name,
                "overall_status": overall_status,
                "overall_score": round(overall_score, 2),
                "shot_type": shot_type,
                
                # InsightFace / archetype
                "archetype_score": round(scores["archetype"].score, 4) if scores["archetype"].score >= 0 else None,
                "insightface_status": scores["archetype"].status,
                "insightface_note": scores["archetype"].note,
                
                # AI Brain
                "ai_brain_score": round(scores["ai_brain"].score, 4) if ("ai_brain" in scores and scores["ai_brain"].score >= 0) else None,
                "ai_brain_status": scores["ai_brain"].status if "ai_brain" in scores else "PASS",
                "ai_brain_note": scores["ai_brain"].note if "ai_brain" in scores else "disabled",
                
                # CLIP semantic
                "semantic_clip_score": round(scores["semantic"].score, 4) if scores["semantic"].score >= 0 else None,
                "semantic_clip_status": scores["semantic"].status,
                "semantic_clip_note": scores["semantic"].note,
                
                # Aesthetic
                "aesthetic_score": round(scores["aesthetic"].score * 10, 2) if scores["aesthetic"].score >= 0 else None,
                "aesthetic_status": scores["aesthetic"].status,
                "aesthetic_note": scores["aesthetic"].note,
                
                # Skin tone
                "skin_tone_score": round(scores["skin"].score, 4) if scores["skin"].score >= 0 else None,
                "skin_tone_status": scores["skin"].status,
                "skin_tone_note": scores["skin"].note,
                
                # Blur
                "blur_score": round(scores["blur"].score, 4) if scores["blur"].score >= 0 else None,
                "blur_status": scores["blur"].status,
                "blur_note": scores["blur"].note,
                
                # Exposure
                "exposure_score": round(scores["exposure"].score, 4) if scores["exposure"].score >= 0 else None,
                "exposure_status": scores["exposure"].status,
                "exposure_note": scores["exposure"].note,
                
                # Pose
                "pose_score": round(scores["pose"].score, 4) if scores["pose"].score >= 0 else None,
                "pose_status": scores["pose"].status,
                "pose_note": scores["pose"].note,
                
                # Duplicate
                "duplicate_status": dup_result.status,
                "duplicate_note": dup_result.note,
                
                # Consolidated reason
                "reason": "; ".join([f"{k}({s.status})" for k, s in scores.items() if s.status != "PASS"]) or "Passed all checks",
            }
            
            record["category"] = self.category_assigner.assign(record, self.cfg)
            self.results.append(record)
            
            # Copy to category folder
            cat_folder = record["category"]
            dest = self.output_dir / cat_folder / img_path.name
            try:
                shutil.copy2(img_path, dest)
            except Exception as e:
                self.logger.error(f"Failed to copy {img_path.name}: {e}")
            
            # Copy to approved_all if not rejected
            if overall_status != "REJECT" and record["category"] != "reject":
                approved_dest = self.output_dir / "approved_all" / img_path.name
                try:
                    shutil.copy2(img_path, approved_dest)
                except Exception:
                    pass
        
        # Write outputs
        self._write_outputs()
        self._print_summary()
    
    def _write_outputs(self):
        # CSV
        if PANDAS_OK and self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values(["category", "overall_score"], ascending=[True, False])
            df.to_csv(self.output_dir / "report.csv", index=False)
        else:
            import csv
            if self.results:
                keys = self.results[0].keys()
                with open(self.output_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
                    dict_writer = csv.DictWriter(f, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(self.results)
        
        # JSON (rich per-image metadata)
        json_data = {
            "config": {
                "character_name": self.cfg.character_name,
                "character_description": self.cfg.character_description,
                "archetype_dir": str(self.archetype_dir),
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "total_images": len(self.results),
            },
            "images": self.results,
        }
        with open(self.output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _print_summary(self):
        if not self.cfg.print_summary:
            return
        
        cat_counts = defaultdict(int)
        status_counts = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
        for r in self.results:
            cat_counts[r["category"]] += 1
            status_counts[r["overall_status"]] += 1
        
        print("\n" + "=" * 60)
        print("  PRODUCTION QC COMPLETE")
        print("=" * 60)
        print(f"\nOverall Status:")
        for s, c in status_counts.items():
            print(f"  {s:8s}: {c}")
        print(f"\nCategory Distribution:")
        for cat, c in sorted(cat_counts.items()):
            print(f"  {cat:20s}: {c}")
        
        print(f"\nTop Face Anchors (by overall score):")
        face_top = sorted([r for r in self.results if r["category"] == "face_anchor"],
                          key=lambda x: x["overall_score"], reverse=True)[:self.cfg.top_n_face_anchors]
        for r in face_top:
            print(f"  {r['filename']:30s} | score={r['overall_score']:5.1f} | arche={r.get('archetype_score', 'N/A'):6s} | aesthetic={r.get('aesthetic_score', 'N/A'):6s}")
        
        print(f"\nTop Full-Body References:")
        body_top = sorted([r for r in self.results if r["category"] == "full_body"],
                          key=lambda x: x["overall_score"], reverse=True)[:self.cfg.top_n_full_body]
        for r in body_top:
            print(f"  {r['filename']:30s} | score={r['overall_score']:5.1f} | pose={r.get('pose_score', 'N/A'):6s}")
        
        print(f"\nFiles written:")
        print(f"  CSV:  {self.output_dir / 'report.csv'}")
        print(f"  JSON: {self.output_dir / 'report.json'}")
        print(f"  Log:  {self.output_dir / 'qc.log'}")
        print("=" * 60)

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pinterest Reference Curation QC — Production AI Edition. "
                    "Filters and categorizes bulk-downloaded images using CLIP, InsightFace, "
                    "and ensemble scoring against a curated archetype."
    )
    parser.add_argument("--config", type=Path, default=Path("qc_config.yaml"),
                        help="Path to YAML configuration file (default: qc_config.yaml)")
    parser.add_argument("--archetype", required=True, type=Path,
                        help="Directory with 5-10 manually curated archetype images.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory with bulk-downloaded Pinterest images.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for categorized reference folders.")
    args = parser.parse_args()
    
    if not args.archetype.exists():
        sys.exit(f"ERROR: Archetype directory not found: {args.archetype}")
    if not args.input.exists():
        sys.exit(f"ERROR: Input directory not found: {args.input}")
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Run pipeline
    pipeline = ProductionPipeline(config, args.archetype, args.input, args.output)
    pipeline.process()


if __name__ == "__main__":
    main()
