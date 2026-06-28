#!/usr/bin/env python3
"""
Pinterest Reference Curation QC Tool
=====================================

Automatically filters, scores, and categorizes bulk-downloaded Pinterest images
against a manually curated archetype. Outputs organized reference folders for
ComfyUI anchor generation and dataset building.

Usage:
------
    1. Manually curate 5-10 perfect archetype images into ./archetype_anchors/
    2. Bulk download Pinterest images into ./pinterest_raw/
    3. Run:
       python pinterest_qc.py --archetype ./archetype_anchors --input ./pinterest_raw --output ./output

Output Structure:
-----------------
    output/
    ├── face_anchors/           # Best close-up face shots (for ComfyUI ReferenceLatent)
    ├── hair_refs/              # Best hair texture/style shots
    ├── outfit_refs/            # Best outfit/clothing shots
    ├── pose_refs/              # Best pose shots (half-body to full-body)
    ├── full_body_refs/         # Best full-body proportion shots
    ├── approved_all/           # All images that passed basic QC
    ├── review/                 # Borderline — manual inspection
    ├── reject/                 # Bad quality or wrong archetype
    └── report.csv              # Full scoring report with category assignments

Dependencies:
-------------
    pip install opencv-python pillow numpy pandas imagehash tqdm torch torchvision facenet-pytorch mediapipe scikit-learn

Tuning:
-------
    All thresholds are in the Thresholds class. For the first run, inspect the
    review/ folder and adjust ARCHE_MATCH_PASS, SKIN_TONE_MAX_DELTA, and shot-type
    thresholds to match your preferences.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

# =============================================================================
# DEPENDENCY RESOLUTION & DEGRADATION
# =============================================================================

try:
    import cv2
    import numpy as np
    from PIL import Image
    CORE_OK = True
except ImportError as e:
    print(f"CRITICAL ERROR: A core dependency is missing: {e}")
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
    import imagehash
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_OK = True
except ImportError:
    torch = None
    FACENET_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Thresholds:
    """Centralized thresholds for Pinterest reference curation."""

    # Archetype face match (cosine similarity to curated anchors)
    ARCHE_MATCH_PASS: float = 0.60      # Good match to archetype
    ARCHE_MATCH_REVIEW: float = 0.35    # Borderline — check manually
    # Below this = reject (wrong person type)

    # Image quality
    BLUR_PASS: float = 100.0
    BLUR_REVIEW: float = 30.0
    CLIP_BLACK_REJECT: float = 25.0
    CLIP_WHITE_REJECT: float = 20.0
    CLIP_REVIEW: float = 5.0

    # Duplicate detection
    DUP_REJECT_DIST: int = 2
    DUP_REVIEW_DIST: int = 8

    # Skin tone consistency (CIEDE2000 in LAB space)
    # Target: warm beige with light olive undertone
    # Anchor skin tone is computed from archetype images
    SKIN_TONE_MAX_DELTA: float = 12.0   # Higher = more tolerant of lighting variation
    SKIN_TONE_REVIEW_DELTA: float = 18.0

    # Shot-type classification (face size as fraction of image)
    CLOSEUP_FACE_MIN: float = 0.35      # Face > 35% of image = close-up
    PORTRAIT_FACE_MIN: float = 0.15     # Face > 15% = portrait
    HALFBODY_FACE_MIN: float = 0.08     # Face > 8% = half-body
    # Below 8% with body visible = full-body
    # No face with body = pose/body ref

    # Body proportions (only checked for full-body / half-body shots)
    SHOULDER_HIP_MIN: float = 0.7
    SHOULDER_HIP_MAX: float = 1.6
    HEAD_BODY_MIN: float = 0.08
    HEAD_BODY_MAX: float = 0.45

    # Attractiveness / quality proxy (combined score floor)
    OVERALL_SCORE_PASS: float = 6.0     # 0-10 scale
    OVERALL_SCORE_REVIEW: float = 4.0

    # Device
    DEVICE: str = "cuda" if (FACENET_OK and torch is not None and torch.cuda.is_available()) else "cpu"


# =============================================================================
# CATEGORIZATION ENGINE
# =============================================================================

class ReferenceCategory:
    """Categories for reference image organization."""
    FACE_ANCHOR = "face_anchor"        # Close-up face, best for ComfyUI ReferenceLatent
    HAIR = "hair"                      # Hair texture/style visible
    OUTFIT = "outfit"                  # Clothing clearly visible
    POSE = "pose"                      # Good pose, partial body visible
    FULL_BODY = "full_body"            # Complete body proportions visible
    APPROVED = "approved"              # Passes basic QC but no specific category
    REVIEW = "review"
    REJECT = "reject"


# =============================================================================
# QC PIPELINE
# =============================================================================

class PinterestReferenceQC:
    def __init__(self, thresholds: Thresholds, archetype_dir: Path):
        self.thresh = thresholds
        self.archetype_dir = archetype_dir
        self.device = torch.device(thresholds.DEVICE) if (FACENET_OK and torch is not None) else None

        self.mtcnn = None
        self.resnet = None
        self.pose_detector = None

        self.archetype_embeddings = None      # (N, 512)
        self.archetype_skin_lab = None          # (3,) mean LAB skin tone
        self.seen_hashes: List[Tuple[Any, Any]] = []
        self.results: List[Dict[str, Any]] = []

        self._init_models()
        self._load_archetype()

    # -------------------------------------------------------------------------
    # Model Init
    # -------------------------------------------------------------------------

    def _init_models(self):
        if not FACENET_OK:
            print("WARNING: facenet-pytorch not available. Face matching disabled.")
        else:
            try:
                print(f"Loading FaceNet on '{self.thresh.DEVICE}'...")
                self.mtcnn = MTCNN(
                    image_size=160, margin=0, min_face_size=20,
                    keep_all=True, device=self.device, post_process=True
                )
                self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            except Exception as e:
                print(f"WARNING: FaceNet load failed: {e}")
                self.mtcnn = None
                self.resnet = None

        if not MEDIAPIPE_OK:
            print("WARNING: mediapipe not available. Pose/body checks disabled.")
        else:
            try:
                print("Loading MediaPipe Pose...")
                self.pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=True, model_complexity=1,
                    min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"WARNING: MediaPipe load failed: {e}")
                self.pose_detector = None

    # -------------------------------------------------------------------------
    # Archetype Loading
    # -------------------------------------------------------------------------

    def _load_archetype(self):
        """Load archetype anchors: compute face embeddings + average skin tone."""
        if not FACENET_OK or self.mtcnn is None or self.resnet is None:
            print("WARNING: FaceNet unavailable. Archetype matching disabled.")
            return

        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = [p for p in self.archetype_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_ext]
        if not paths:
            print(f"WARNING: No archetype images in {self.archetype_dir}. Matching disabled.")
            return

        embs = []
        skin_labs = []

        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    # Face embedding
                    faces = self.mtcnn(img)
                    if faces is None:
                        print(f"WARNING: No face in archetype '{p.name}'")
                        continue
                    if faces.ndim == 3:
                        faces = faces.unsqueeze(0)
                    # Use only the largest / primary face
                    emb = self.resnet(faces[0:1].to(self.device))
                    embs.append(emb)

                    # Skin tone from the same image (using face crop if possible)
                    cv_img = cv2.imread(str(p))
                    if cv_img is not None:
                        skin_lab = self._extract_skin_tone(cv_img, use_center=True)
                        if skin_lab is not None:
                            skin_labs.append(skin_lab)
                except Exception as e:
                    print(f"WARNING: Failed archetype '{p.name}': {e}")

        if embs:
            self.archetype_embeddings = torch.cat(embs, dim=0)
            print(f"Loaded {len(embs)} archetype face embeddings.")
        if skin_labs:
            self.archetype_skin_lab = np.mean(skin_labs, axis=0)
            print(f"Archetype skin tone LAB: {self.archetype_skin_lab}")

    # -------------------------------------------------------------------------
    # Skin Tone Analysis (CIELAB)
    # -------------------------------------------------------------------------

    def _extract_skin_tone(self, cv_img: np.ndarray, use_center: bool = False) -> Optional[np.ndarray]:
        """Extract average skin tone in CIELAB space. Returns (L, A, B) or None."""
        try:
            if use_center:
                # For archetype images, use center crop (usually contains face)
                h, w = cv_img.shape[:2]
                cx, cy = w // 2, h // 2
                crop = cv_img[max(0, cy - h//3):min(h, cy + h//3), max(0, cx - w//3):min(w, cx + w//3)]
            else:
                # For candidate images, use simple skin color threshold in YCrCb
                crop = cv_img

            # Convert to YCrCb for skin detection
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            # Cr channel: skin pixels roughly 140-180, Cb: 90-130
            mask = (
                (ycrcb[:, :, 1] >= 135) & (ycrcb[:, :, 1] <= 180) &
                (ycrcb[:, :, 2] >= 85) & (ycrcb[:, :, 2] <= 135)
            )
            skin_pixels = crop[mask]
            if len(skin_pixels) < 50:
                return None

            # Convert skin pixels to LAB
            lab_pixels = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            avg_lab = np.mean(lab_pixels, axis=0).astype(np.float32)
            return avg_lab
        except Exception:
            return None

    def _delta_e_ciede2000(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Simplified CIEDE2000 approximation. Returns perceptual color distance."""
        # Fast Euclidean in LAB is ~90% accurate for small deltas; use full formula if needed
        return float(np.linalg.norm(lab1 - lab2))

    # -------------------------------------------------------------------------
    # Shot Type Classification
    # -------------------------------------------------------------------------

    def classify_shot_type(self, cv_img: np.ndarray, face_box: Optional[Tuple]) -> str:
        """Classify image by shot type based on face size and body visibility."""
        h, w = cv_img.shape[:2]
        img_area = h * w

        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_area = fw * fh
            face_ratio = face_area / img_area

            if face_ratio >= self.thresh.CLOSEUP_FACE_MIN:
                return "closeup"
            if face_ratio >= self.thresh.PORTRAIT_FACE_MIN:
                return "portrait"
            if face_ratio >= self.thresh.HALFBODY_FACE_MIN:
                return "half_body"
            # Face is small but present — check if full body is visible
            if self._has_full_body(cv_img):
                return "full_body"
            return "environmental"

        # No face detected — check for body
        if self._has_full_body(cv_img):
            return "full_body"
        return "environmental"

    def _has_full_body(self, cv_img: np.ndarray) -> bool:
        """Check if pose landmarks indicate a full body is visible."""
        if not MEDIAPIPE_OK or self.pose_detector is None:
            return False
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb)
            if not results.pose_landmarks:
                return False
            lm = results.pose_landmarks.landmark
            # Full body = visible shoulders, hips, and at least one knee/ankle
            has_shoulders = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5
            has_hips = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].visibility > 0.5
            has_legs = (
                lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.5 or
                lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.5
            )
            return has_shoulders and has_hips and has_legs
        except Exception:
            return False

    def _get_face_box(self, cv_img: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Get primary face bounding box as (x, y, w, h) normalized to image size."""
        if not FACENET_OK or self.mtcnn is None:
            return None
        try:
            pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            faces = self.mtcnn(pil)
            if faces is None:
                return None
            # MTCNN also returns bounding boxes if we call it with return_prob=False
            # Re-detect to get boxes
            boxes, _ = self.mtcnn.detect(pil)
            if boxes is None or len(boxes) == 0:
                return None
            # Use largest face
            h_img, w_img = cv_img.shape[:2]
            largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            x, y, x2, y2 = largest
            return (x / w_img, y / h_img, (x2-x) / w_img, (y2-y) / h_img)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # QC Scorers (adapted from lora_qc.py)
    # -------------------------------------------------------------------------

    def score_archetype_match(self, pil_img: Optional[Image.Image]) -> Tuple[float, str, str]:
        if not FACENET_OK or self.mtcnn is None or self.resnet is None:
            return -1.0, "PASS", "facenet_unavailable"
        if self.archetype_embeddings is None:
            return -1.0, "PASS", "archetype_missing"
        if pil_img is None:
            return 0.0, "REJECT", "image_missing"

        try:
            with torch.no_grad():
                faces = self.mtcnn(pil_img)
                if faces is None:
                    return 0.0, "REJECT", "no_face"
                if faces.ndim == 3:
                    faces = faces.unsqueeze(0)
                embs = self.resnet(faces.to(self.device))
                sims = torch.nn.functional.cosine_similarity(
                    embs.unsqueeze(1),
                    self.archetype_embeddings.unsqueeze(0),
                    dim=2,
                )
                best = sims.max().item()

            if best >= self.thresh.ARCHE_MATCH_PASS:
                return best, "PASS", f"match={best:.3f}"
            if best >= self.thresh.ARCHE_MATCH_REVIEW:
                return best, "REVIEW", f"match={best:.3f}"
            return best, "REJECT", f"match={best:.3f}"
        except Exception as e:
            return -1.0, "REVIEW", f"archetype_error:{e}"

    def score_skin_tone(self, cv_img: Optional[np.ndarray]) -> Tuple[float, str, str]:
        if self.archetype_skin_lab is None or cv_img is None:
            return -1.0, "PASS", "skin_tone_unavailable"
        try:
            candidate_lab = self._extract_skin_tone(cv_img, use_center=False)
            if candidate_lab is None:
                return -1.0, "REVIEW", "no_skin_detected"
            delta = self._delta_e_ciede2000(self.archetype_skin_lab, candidate_lab)
            if delta <= self.thresh.SKIN_TONE_MAX_DELTA:
                return delta, "PASS", f"delta_e={delta:.1f}"
            if delta <= self.thresh.SKIN_TONE_REVIEW_DELTA:
                return delta, "REVIEW", f"delta_e={delta:.1f}"
            return delta, "REJECT", f"delta_e={delta:.1f}"
        except Exception as e:
            return -1.0, "REVIEW", f"skin_tone_error:{e}"

    def score_blur(self, cv_img: Optional[np.ndarray]) -> Tuple[float, str, str]:
        if cv_img is None:
            return 0.0, "REVIEW", "image_missing"
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            return 0.0, "REVIEW", f"blur_error:{e}"
        if score >= self.thresh.BLUR_PASS:
            return score, "PASS", f"lapvar={score:.1f}"
        if score >= self.thresh.BLUR_REVIEW:
            return score, "REVIEW", f"lapvar={score:.1f}"
        return score, "REJECT", f"lapvar={score:.1f}"

    def score_exposure(self, cv_img: Optional[np.ndarray]) -> Tuple[Tuple[float, float], str, str]:
        if cv_img is None:
            return (0.0, 0.0), "REVIEW", "image_missing"
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            total = gray.size
            black = float(np.sum(gray < 10) / total * 100)
            white = float(np.sum(gray > 245) / total * 100)
        except Exception as e:
            return (0.0, 0.0), "REVIEW", f"exposure_error:{e}"
        if black > self.thresh.CLIP_BLACK_REJECT or white > self.thresh.CLIP_WHITE_REJECT:
            return (black, white), "REJECT", f"black={black:.1f}% white={white:.1f}%"
        if black > self.thresh.CLIP_REVIEW or white > self.thresh.CLIP_REVIEW:
            return (black, white), "REVIEW", f"black={black:.1f}% white={white:.1f}%"
        return (black, white), "PASS", f"black={black:.1f}% white={white:.1f}%"

    def score_duplicate(self, pil_img: Optional[Image.Image]) -> Tuple[bool, str, str]:
        if not IMAGEHASH_OK or pil_img is None:
            return False, "PASS", "imagehash_unavailable"
        try:
            phash = imagehash.phash(pil_img)
            dhash = imagehash.dhash(pil_img)
        except Exception as e:
            return False, "PASS", f"hash_error:{e}"
        for prev_phash, prev_dhash in self.seen_hashes:
            try:
                pdist = phash - prev_phash
                ddist = dhash - prev_dhash
                if pdist <= self.thresh.DUP_REJECT_DIST and ddist <= self.thresh.DUP_REJECT_DIST:
                    return True, "REJECT", f"dup_p:{pdist}_d:{ddist}"
                if pdist <= self.thresh.DUP_REVIEW_DIST and ddist <= self.thresh.DUP_REVIEW_DIST:
                    self.seen_hashes.append((phash, dhash))
                    return True, "REVIEW", f"near_dup_p:{pdist}_d:{ddist}"
            except Exception:
                continue
        self.seen_hashes.append((phash, dhash))
        return False, "PASS", "unique"

    def score_pose(self, cv_img: Optional[np.ndarray]) -> Tuple[Dict[str, float], str, str]:
        if not MEDIAPIPE_OK or self.pose_detector is None:
            return {}, "PASS", "mediapipe_unavailable"
        if cv_img is None:
            return {}, "REVIEW", "image_missing"
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb)
        except Exception as e:
            return {}, "REVIEW", f"pose_error:{e}"
        if not results.pose_landmarks:
            return {}, "REVIEW", "no_pose"
        try:
            lm = results.pose_landmarks.landmark
            def get(name):
                return lm[mp.solutions.pose.PoseLandmark[name].value]
            key_points = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
            if any(get(k).visibility < 0.5 for k in key_points):
                return {}, "PASS", "partial_pose"
            shoulder_w = abs(get("LEFT_SHOULDER").x - get("RIGHT_SHOULDER").x)
            hip_w = abs(get("LEFT_HIP").x - get("RIGHT_HIP").x)
            sh_ratio = shoulder_w / hip_w if hip_w > 0 else 0
            head_h = abs(get("NOSE").y - (get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2)
            body_h = abs((get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2 -
                         (get("LEFT_HIP").y + get("RIGHT_HIP").y) / 2)
            hb_ratio = head_h / body_h if body_h > 0 else 0
        except Exception as e:
            return {}, "REVIEW", f"pose_proc_error:{e}"
        metrics = {"sh_ratio": sh_ratio, "hb_ratio": hb_ratio}
        notes = []
        status = "PASS"
        if not (self.thresh.SHOULDER_HIP_MIN <= sh_ratio <= self.thresh.SHOULDER_HIP_MAX):
            notes.append(f"sh_ratio={sh_ratio:.2f}")
            status = "REVIEW"
        if not (self.thresh.HEAD_BODY_MIN <= hb_ratio <= self.thresh.HEAD_BODY_MAX):
            notes.append(f"hb_ratio={hb_ratio:.2f}")
            status = "REVIEW"
        return metrics, status, "; ".join(notes) if notes else "proportions_ok"

    # -------------------------------------------------------------------------
    # Category Assignment Logic
    # -------------------------------------------------------------------------

    def assign_category(self, record: Dict[str, Any]) -> str:
        """Assign the best reference category for this image."""
        if record["overall_status"] == "REJECT":
            return ReferenceCategory.REJECT

        shot = record["shot_type"]
        arche_status = record["archetype_status"]
        skin_status = record["skin_tone_status"]

        # Must have at least archetype or skin tone pass to be a strong reference
        good_match = (arche_status == "PASS" or skin_status == "PASS")

        if not good_match and record["overall_status"] == "REVIEW":
            return ReferenceCategory.REVIEW

        # Close-up face → primary anchor for ComfyUI ReferenceLatent
        if shot == "closeup" and arche_status == "PASS":
            return ReferenceCategory.FACE_ANCHOR

        # Portrait with good face → secondary face ref or hair ref
        if shot == "portrait":
            if arche_status == "PASS":
                # Could be hair or face — use as hair if outfit is also visible
                if record["pose_status"] == "PASS" and record["blur_status"] == "PASS":
                    return ReferenceCategory.HAIR
                return ReferenceCategory.FACE_ANCHOR
            return ReferenceCategory.HAIR

        # Half-body → outfit or pose
        if shot == "half_body":
            if arche_status == "PASS" and record["pose_status"] == "PASS":
                return ReferenceCategory.OUTFIT
            return ReferenceCategory.POSE

        # Full-body → full body proportions
        if shot == "full_body":
            if record["pose_status"] == "PASS":
                return ReferenceCategory.FULL_BODY
            return ReferenceCategory.POSE

        # Environmental or no clear category → approved if it passes
        if record["overall_status"] == "PASS":
            return ReferenceCategory.APPROVED

        return ReferenceCategory.REVIEW

    # -------------------------------------------------------------------------
    # Overall Scoring
    # -------------------------------------------------------------------------

    def compute_overall_score(self, record: Dict[str, Any]) -> float:
        """Compute a 0-10 overall quality score for ranking within categories."""
        score = 5.0

        # Archetype match (0-3 points)
        arche_score = record.get("archetype_score", 0)
        if arche_score >= 0:
            score += min(3.0, arche_score * 3.0)
        else:
            score -= 1.0

        # Skin tone match (0-2 points)
        skin_delta = record.get("skin_tone_delta", 999)
        if skin_delta >= 0:
            if skin_delta <= self.thresh.SKIN_TONE_MAX_DELTA:
                score += 2.0 * (1 - skin_delta / self.thresh.SKIN_TONE_MAX_DELTA)
            else:
                score -= 1.0

        # Image quality (0-2 points)
        blur = record.get("blur_score", 0)
        if blur >= self.thresh.BLUR_PASS:
            score += 2.0
        elif blur >= self.thresh.BLUR_REVIEW:
            score += 1.0
        else:
            score -= 1.0

        # Pose quality (0-1.5 points)
        if record.get("pose_status") == "PASS":
            score += 1.5
        elif record.get("pose_status") == "REVIEW":
            score += 0.5

        # Exposure (0-1.5 points)
        if record.get("exposure_status") == "PASS":
            score += 1.5

        return max(0.0, min(10.0, score))

    def judge_status(self, scores: Dict[str, str]) -> str:
        statuses = [scores["archetype"], scores["blur"], scores["exposure"], scores["dup"], scores["skin_tone"]]
        if "REJECT" in statuses:
            return "REJECT"
        if "REVIEW" in statuses:
            return "REVIEW"
        # For pose, only reject if full body and pose is bad
        if scores["pose"] == "REJECT":
            return "REVIEW"
        return "PASS"

    # -------------------------------------------------------------------------
    # Pipeline Runner
    # -------------------------------------------------------------------------

    def process(self, input_dir: Path, output_dir: Path):
        # Create output folders
        for sub in ("face_anchors", "hair_refs", "outfit_refs", "pose_refs",
                    "full_body_refs", "approved_all", "review", "reject"):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_paths = sorted([p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_ext])
        if not image_paths:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(image_paths)} Pinterest images against archetype...")

        for img_path in tqdm(image_paths):
            # Load image
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                pil_img = None
                print(f"WARNING: Pillow failed on '{img_path.name}': {e}")

            try:
                cv_img = cv2.imread(str(img_path))
                if cv_img is None:
                    raise ValueError("cv2.imread returned None")
            except Exception as e:
                cv_img = None
                print(f"WARNING: OpenCV failed on '{img_path.name}': {e}")

            if pil_img is None and cv_img is None:
                print(f"ERROR: Cannot load '{img_path.name}'. Skipping.")
                continue

            # Run all checkers
            arche_score, arche_status, arche_note = self.score_archetype_match(pil_img)
            skin_delta, skin_status, skin_note = self.score_skin_tone(cv_img)
            blur_score, blur_status, blur_note = self.score_blur(cv_img)
            (black_pct, white_pct), exp_status, exp_note = self.score_exposure(cv_img)
            _, dup_status, dup_note = self.score_duplicate(pil_img)
            pose_metrics, pose_status, pose_note = self.score_pose(cv_img)

            # Shot type
            face_box = self._get_face_box(cv_img) if cv_img is not None else None
            shot_type = self.classify_shot_type(cv_img, face_box) if cv_img is not None else "unknown"

            # Build record
            overall_status = self.judge_status({
                "archetype": arche_status,
                "blur": blur_status,
                "exposure": exp_status,
                "dup": dup_status,
                "skin_tone": skin_status,
                "pose": pose_status,
            })

            record = {
                "filename": img_path.name,
                "overall_status": overall_status,
                "archetype_score": round(arche_score, 4) if arche_score >= 0 else None,
                "archetype_status": arche_status,
                "archetype_note": arche_note,
                "skin_tone_delta": round(skin_delta, 2) if skin_delta >= 0 else None,
                "skin_tone_status": skin_status,
                "skin_tone_note": skin_note,
                "blur_score": round(blur_score, 2),
                "blur_status": blur_status,
                "blur_note": blur_note,
                "exposure_black_pct": round(black_pct, 2),
                "exposure_white_pct": round(white_pct, 2),
                "exposure_status": exp_status,
                "exposure_note": exp_note,
                "duplicate_status": dup_status,
                "duplicate_note": dup_note,
                "pose_status": pose_status,
                "pose_note": pose_note,
                "pose_shoulder_hip_ratio": round(pose_metrics.get("sh_ratio", -1), 3),
                "pose_head_body_ratio": round(pose_metrics.get("hb_ratio", -1), 3),
                "shot_type": shot_type,
            }

            record["overall_score"] = round(self.compute_overall_score(record), 2)
            record["category"] = self.assign_category(record)

            # Consolidate reason
            reasons = []
            for gate, status in [
                ("Archetype", arche_status), ("SkinTone", skin_status),
                ("Blur", blur_status), ("Exposure", exp_status),
                ("Duplicate", dup_status), ("Pose", pose_status)
            ]:
                if status != "PASS":
                    reasons.append(f"{gate}({status})")
            record["reason"] = "; ".join(reasons) if reasons else "Passed all checks"

            self.results.append(record)

            # Copy to category folder + overall status folder
            cat_folder = record["category"]
            dest = output_dir / cat_folder / img_path.name
            try:
                shutil.copy2(img_path, dest)
            except Exception as e:
                print(f"ERROR: Failed to copy '{img_path.name}': {e}")

            # Also copy to approved_all if PASS or certain REVIEW categories
            if overall_status in ("PASS", "REVIEW") and record["category"] != ReferenceCategory.REJECT:
                approved_dest = output_dir / "approved_all" / img_path.name
                try:
                    shutil.copy2(img_path, approved_dest)
                except Exception:
                    pass

        # Write report.csv
        csv_path = output_dir / "report.csv"
        if PANDAS_OK and self.results:
            df = pd.DataFrame(self.results)
            # Sort by category then by overall score descending
            df = df.sort_values(["category", "overall_score"], ascending=[True, False])
            df.to_csv(csv_path, index=False)
        else:
            import csv
            if self.results:
                keys = self.results[0].keys()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    dict_writer = csv.DictWriter(f, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(self.results)

        # Print summary
        cat_counts = defaultdict(int)
        status_counts = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
        for r in self.results:
            cat_counts[r["category"]] += 1
            status_counts[r["overall_status"]] += 1

        print(f"\nDone. Report: {csv_path}")
        print(f"\nOverall Status:")
        for s, c in status_counts.items():
            print(f"  {s}: {c}")
        print(f"\nCategory Distribution:")
        for cat, c in sorted(cat_counts.items()):
            print(f"  {cat}: {c}")
        print(f"\nTop face_anchors by score:")
        face_top = sorted([r for r in self.results if r["category"] == ReferenceCategory.FACE_ANCHOR],
                          key=lambda x: x["overall_score"], reverse=True)[:5]
        for r in face_top:
            print(f"    {r['filename']} — score={r['overall_score']}, arche_match={r['archetype_score']}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pinterest Reference Curation QC. Filters bulk-downloaded images against a curated archetype."
    )
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

    thresholds = Thresholds()
    pipeline = PinterestReferenceQC(thresholds, args.archetype)
    pipeline.process(args.input, args.output)


if __name__ == "__main__":
    main()
