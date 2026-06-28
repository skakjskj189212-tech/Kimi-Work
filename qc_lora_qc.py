#!/usr/bin/env python3
"""
LoRA Dataset QC Tool
====================

A pilot local Quality Control (QC) tool for filtering AI-generated character images
prior to LoRA training. It classifies candidate images into PASS, REVIEW, and REJECT categories
based on multiple computer vision metrics:
- Face identity consistency (using FaceNet cosine similarity against reference anchors)
- Image sharpness/blur (using OpenCV Laplacian variance)
- Over/under-exposure clipping (using OpenCV pixel intensity histograms)
- Duplicates or near-duplicates (using imagehash perceptual/difference hashing)
- Pose and body proportion anomalies (using MediaPipe Pose landmarks)
- Hand artifact and structure warnings (using MediaPipe Hands)

Folder Structure:
-----------------
.
├── anchors/               # Put 1 or more high-quality reference/anchor images here
├── input/                 # Put your generated/candidate images here
└── output/                # QC results will be saved here:
    ├── pass/              # Images that passed all checks (ready for LoRA training)
    ├── review/            # Borderline/uncertain images (manually inspect these!)
    ├── reject/            # Definite failures (blurry, duplicated, badly clipped)
    └── report.csv         # Detailed scoring report with metrics and decision notes

Installation:
-------------
Install the required packages. Optional heavy models degrade gracefully if not installed:

    pip install opencv-python pillow numpy pandas imagehash tqdm torch torchvision facenet-pytorch mediapipe

Running the Tool:
-----------------
Execute from your terminal:

    python lora_qc.py --anchors ./anchors --input ./input --output ./output

Tuning Workflow:
----------------
All thresholds are centralized in the `Thresholds` class. For the pilot run,
review the borderline cases in the `output/review` folder, compare their scores
in `report.csv`, and adjust thresholds accordingly:
- `IDENTITY_PASS` & `IDENTITY_REVIEW`: Raise if identity is too loose; lower if it rejects too many good matches.
- `BLUR_PASS` & `BLUR_REVIEW`: Adjust based on your dataset's baseline style (e.g. digital art vs. photo).
- `DUP_REJECT_DIST` & `DUP_REVIEW_DIST`: Adjust if near-duplicates or similar crops are passing through.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

# =============================================================================
# DEPENDENCY RESOLUTION & DEGRADATION
# =============================================================================

# Core libraries - wrapped with warning but required for basic operation
try:
    import cv2
    import numpy as np
    from PIL import Image
    CORE_OK = True
except ImportError as e:
    print(f"CRITICAL ERROR: A core dependency is missing: {e}")
    print("Please install required packages:")
    print("  pip install opencv-python pillow numpy")
    sys.exit(1)

# Optional helper: pandas
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# Optional helper: tqdm
try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False
    # Simple fallback iterator
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Optional helper: imagehash
try:
    import imagehash
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

# Optional heavy model libraries
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


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Thresholds:
    """Centralized thresholds. Tune these based on your manual borderline review."""

    # Face identity consistency (cosine similarity, 0–1)
    IDENTITY_PASS: float = 0.70
    IDENTITY_REVIEW: float = 0.45

    # Blur (Laplacian variance, higher is sharper)
    BLUR_PASS: float = 120.0
    BLUR_REVIEW: float = 30.0

    # Exposure (percentage of pixels in clipping tails)
    CLIP_BLACK_REJECT: float = 20.0  # % near 0 (prevent false rejects on dark hair/backgrounds)
    CLIP_WHITE_REJECT: float = 15.0  # % near 255 (blown out highlights)
    CLIP_REVIEW: float = 5.0

    # Duplicate detection (perceptual hash distance, lower = more similar)
    DUP_REJECT_DIST: int = 2         # exact / near-exact
    DUP_REVIEW_DIST: int = 8         # suspiciously similar

    # Body proportions (MediaPipe heuristics; wide ranges by default)
    SHOULDER_HIP_MIN: float = 0.7
    SHOULDER_HIP_MAX: float = 1.6
    HEAD_BODY_MIN: float = 0.08
    HEAD_BODY_MAX: float = 0.45

    # Hand detection confidence
    HAND_CONF_REVIEW: float = 0.6

    # Torch device selection
    DEVICE: str = "cuda" if (FACENET_OK and torch is not None and torch.cuda.is_available()) else "cpu"


# =============================================================================
# QC PIPELINE
# =============================================================================

class LoRAQCPipeline:
    def __init__(self, thresholds: Thresholds, anchor_dir: Path):
        self.thresh = thresholds
        self.anchor_dir = anchor_dir
        self.device = torch.device(thresholds.DEVICE) if (FACENET_OK and torch is not None) else None

        self.mtcnn = None
        self.resnet = None
        self.pose_detector = None
        self.hands_detector = None

        self.anchor_embeddings = None          # Shape: (N, 512)
        self.seen_hashes: List[Tuple[Any, Any]] = []  # List of (phash, dhash)
        self.results: List[Dict[str, Any]] = []

        self._init_models()
        self._load_anchors()

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------

    def _init_models(self):
        if not FACENET_OK:
            print("WARNING: facenet-pytorch/torch not available. Identity checks disabled.")
        else:
            try:
                print(f"Loading FaceNet models on '{self.thresh.DEVICE}'...")
                self.mtcnn = MTCNN(
                    image_size=160, margin=0, min_face_size=20,
                    keep_all=True, device=self.device, post_process=True
                )
                self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            except Exception as e:
                print(f"WARNING: Failed to load FaceNet models: {e}. Identity checks disabled.")
                self.mtcnn = None
                self.resnet = None

        if not MEDIAPIPE_OK:
            print("WARNING: mediapipe not available. Pose/hand checks disabled.")
        else:
            try:
                print("Loading MediaPipe models...")
                self.pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=True, model_complexity=1,
                    min_detection_confidence=0.5
                )
                self.hands_detector = mp.solutions.hands.Hands(
                    static_image_mode=True, max_num_hands=2,
                    min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"WARNING: Failed to load MediaPipe models: {e}. Pose/hand checks disabled.")
                self.pose_detector = None
                self.hands_detector = None

    def _load_anchors(self):
        if not FACENET_OK or self.mtcnn is None or self.resnet is None:
            return

        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = [
            p for p in self.anchor_dir.glob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
        if not paths:
            print(f"WARNING: No anchor images found in {self.anchor_dir}. Identity checks disabled.")
            return

        embs = []
        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    # Detect face (returns tensor or list/None depending on MTCNN call)
                    face = self.mtcnn(img)
                    if face is None:
                        print(f"WARNING: No face detected in anchor '{p.name}'")
                        continue
                    if face.ndim == 3:
                        face = face.unsqueeze(0)
                    emb = self.resnet(face.to(self.device))
                    embs.append(emb)
                except Exception as e:
                    print(f"WARNING: Failed to process anchor '{p.name}': {e}")
                    continue

        if not embs:
            print("WARNING: No valid anchor faces found. Identity checks disabled.")
            return
        
        self.anchor_embeddings = torch.cat(embs, dim=0)
        print(f"Loaded {len(embs)} anchor face embeddings.")

    # -------------------------------------------------------------------------
    # QC Check Scorers
    # -------------------------------------------------------------------------

    def score_identity(self, pil_img: Optional[Image.Image]) -> Tuple[float, str, str]:
        if not FACENET_OK or self.mtcnn is None or self.resnet is None:
            return -1.0, "PASS", "facenet_unavailable"
        if self.anchor_embeddings is None:
            return -1.0, "PASS", "anchors_missing"
        if pil_img is None:
            return -1.0, "REVIEW", "pil_image_missing"

        try:
            with torch.no_grad():
                faces = self.mtcnn(pil_img)
                if faces is None:
                    return 0.0, "REVIEW", "no_face"

                if faces.ndim == 3:
                    faces = faces.unsqueeze(0)

                num_detected_faces = faces.shape[0]
                embs = self.resnet(faces.to(self.device))
                sims = torch.nn.functional.cosine_similarity(
                    embs.unsqueeze(1),
                    self.anchor_embeddings.unsqueeze(0),
                    dim=2,
                )
                best = sims.max().item()

            if num_detected_faces > 1:
                return best, "REVIEW", f"multiple_faces_detected:{num_detected_faces}; match={best:.3f}"

            if best >= self.thresh.IDENTITY_PASS:
                return best, "PASS", f"match={best:.3f}"
            
            # Low identity similarity scores are routed to REVIEW rather than REJECT.
            # This prevents false rejects on profile shots, extreme angles, or costume changes.
            return best, "REVIEW", f"match={best:.3f}"

        except Exception as e:
            return -1.0, "REVIEW", f"identity_error:{e}"

    def score_blur(self, cv_img: Optional[np.ndarray]) -> Tuple[float, str, str]:
        if cv_img is None:
            return 0.0, "REVIEW", "opencv_image_missing"
        
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
            return (0.0, 0.0), "REVIEW", "opencv_image_missing"
        
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

        is_dup = False
        status = "PASS"
        note = "unique"

        for prev_phash, prev_dhash in self.seen_hashes:
            try:
                pdist = phash - prev_phash
                ddist = dhash - prev_dhash
                if pdist <= self.thresh.DUP_REJECT_DIST and ddist <= self.thresh.DUP_REJECT_DIST:
                    return True, "REJECT", f"dup_p:{pdist}_d:{ddist}"
                if pdist <= self.thresh.DUP_REVIEW_DIST and ddist <= self.thresh.DUP_REVIEW_DIST:
                    is_dup = True
                    status = "REVIEW"
                    note = f"near_dup_p:{pdist}_d:{ddist}"
            except Exception:
                continue

        self.seen_hashes.append((phash, dhash))
        return is_dup, status, note

    def score_pose(self, cv_img: Optional[np.ndarray]) -> Tuple[Dict[str, float], str, str]:
        if not MEDIAPIPE_OK or self.pose_detector is None:
            return {}, "PASS", "mediapipe_unavailable"
        if cv_img is None:
            return {}, "REVIEW", "cv_image_missing"

        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb)
        except Exception as e:
            return {}, "REVIEW", f"pose_error:{e}"

        if not results.pose_landmarks:
            return {}, "REVIEW", "no_pose_landmarks"

        try:
            lm = results.pose_landmarks.landmark

            def get(name: str):
                return lm[mp.solutions.pose.PoseLandmark[name].value]

            key_points = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
            if any(get(k).visibility < 0.5 for k in key_points):
                return {}, "PASS", "partial_pose_or_portrait"

            shoulder_w = abs(get("LEFT_SHOULDER").x - get("RIGHT_SHOULDER").x)
            hip_w = abs(get("LEFT_HIP").x - get("RIGHT_HIP").x)
            sh_ratio = shoulder_w / hip_w if hip_w > 0 else 0

            head_h = abs(get("NOSE").y - (get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2)
            body_h = abs((get("LEFT_SHOULDER").y + get("RIGHT_SHOULDER").y) / 2 -
                         (get("LEFT_HIP").y + get("RIGHT_HIP").y) / 2)
            hb_ratio = head_h / body_h if body_h > 0 else 0
        except Exception as e:
            return {}, "REVIEW", f"pose_processing_error:{e}"

        metrics = {"sh_ratio": sh_ratio, "hb_ratio": hb_ratio}
        notes = []
        status = "PASS"

        if not (self.thresh.SHOULDER_HIP_MIN <= sh_ratio <= self.thresh.SHOULDER_HIP_MAX):
            notes.append(f"shoulder_hip_ratio={sh_ratio:.2f}")
            status = "REVIEW"
        if not (self.thresh.HEAD_BODY_MIN <= hb_ratio <= self.thresh.HEAD_BODY_MAX):
            notes.append(f"head_body_ratio={hb_ratio:.2f}")
            status = "REVIEW"

        return metrics, status, "; ".join(notes) if notes else "proportions_ok"

    def score_hands(self, cv_img: Optional[np.ndarray]) -> Tuple[Dict[str, Any], str, str]:
        if not MEDIAPIPE_OK or self.hands_detector is None:
            return {}, "PASS", "mediapipe_unavailable"
        if cv_img is None:
            return {}, "REVIEW", "cv_image_missing"

        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb)
        except Exception as e:
            return {}, "REVIEW", f"hands_error:{e}"

        if not results.multi_hand_landmarks:
            return {}, "PASS", "no_hands_detected"

        try:
            num_hands = len(results.multi_hand_landmarks)
            confs = [h.classification[0].score for h in results.multi_handedness]
            avg_conf = float(np.mean(confs))

            if avg_conf < self.thresh.HAND_CONF_REVIEW:
                return {"num_hands": num_hands, "conf": avg_conf}, "REVIEW", f"low_conf_hands={avg_conf:.2f}"

            # Heuristic: suspiciously clustered landmarks (indicating a hand shape anomaly)
            for hand in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                # Only check clustering if the hand is large enough in frame to avoid false positives on distant hands
                if bbox_w > 0.08 or bbox_h > 0.08:
                    if np.std(xs) + np.std(ys) < 0.05:
                        return {"num_hands": num_hands, "conf": avg_conf}, "REVIEW", "clustered_hand_landmarks"
        except Exception as e:
            return {}, "REVIEW", f"hands_processing_error:{e}"

        return {"num_hands": num_hands, "conf": avg_conf}, "PASS", f"hands_ok={num_hands}"

    # -------------------------------------------------------------------------
    # Aggregation & Decision Logic
    # -------------------------------------------------------------------------

    def judge(self, scores: Dict[str, str]) -> str:
        statuses = [
            scores["identity_status"],
            scores["blur_status"],
            scores["exposure_status"],
            scores["dup_status"],
            scores["pose_status"],
            scores["hand_status"],
        ]
        if "REJECT" in statuses:
            return "REJECT"
        if "REVIEW" in statuses:
            return "REVIEW"
        return "PASS"

    # -------------------------------------------------------------------------
    # Pipeline Runner
    # -------------------------------------------------------------------------

    def process(self, input_dir: Path, output_dir: Path):
        # Create output folders automatically
        for sub in ("pass", "review", "reject"):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_paths = sorted([
            p for p in input_dir.glob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])
        if not image_paths:
            print(f"No candidate images found in {input_dir}")
            return

        print(f"Processing {len(image_paths)} images...")

        for img_path in tqdm(image_paths):
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                pil_img = None
                print(f"WARNING: Pillow failed to open '{img_path.name}': {e}")

            try:
                cv_img = cv2.imread(str(img_path))
                if cv_img is None:
                    raise ValueError("cv2.imread returned None")
            except Exception as e:
                cv_img = None
                print(f"WARNING: OpenCV failed to read '{img_path.name}': {e}")

            if pil_img is None and cv_img is None:
                print(f"ERROR: Cannot load '{img_path.name}' with Pillow or OpenCV. Skipping.")
                continue

            # Run all checkers
            id_score, id_status, id_note = self.score_identity(pil_img)
            blur_score, blur_status, blur_note = self.score_blur(cv_img)
            (black_pct, white_pct), exp_status, exp_note = self.score_exposure(cv_img)
            _, dup_status, dup_note = self.score_duplicate(pil_img)
            pose_metrics, pose_status, pose_note = self.score_pose(cv_img)
            hand_metrics, hand_status, hand_note = self.score_hands(cv_img)

            # Consolidated reason notes for CSV report and diagnostic review
            reasons = []
            if id_status != "PASS":
                reasons.append(f"Identity ({id_status}): {id_note}")
            if blur_status != "PASS":
                reasons.append(f"Blur ({blur_status}): {blur_note}")
            if exp_status != "PASS":
                reasons.append(f"Exposure ({exp_status}): {exp_note}")
            if dup_status != "PASS":
                reasons.append(f"Duplicate ({dup_status}): {dup_note}")
            if pose_status != "PASS":
                reasons.append(f"Pose ({pose_status}): {pose_note}")
            if hand_status != "PASS":
                reasons.append(f"Hand ({hand_status}): {hand_note}")
            overall_reason = "; ".join(reasons) if reasons else "Passed all checks"

            # Create entry dictionary
            record = {
                "filename": img_path.name,
                "overall_status": self.judge({
                    "identity_status": id_status,
                    "blur_status": blur_status,
                    "exposure_status": exp_status,
                    "dup_status": dup_status,
                    "pose_status": pose_status,
                    "hand_status": hand_status,
                }),
                "reason": overall_reason,
                "identity_score": round(id_score, 4) if id_score >= 0 else None,
                "identity_status": id_status,
                "identity_note": id_note,
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
                "hand_status": hand_status,
                "hand_note": hand_note,
                "hand_num": hand_metrics.get("num_hands", 0),
                "hand_conf": round(hand_metrics.get("conf", 0), 3),
            }

            self.results.append(record)

            # Copy image to designated output folder, preserving filename
            dest = output_dir / record["overall_status"].lower() / img_path.name
            try:
                shutil.copy2(img_path, dest)
            except Exception as e:
                print(f"ERROR: Failed to copy '{img_path.name}' to '{dest}': {e}")

        # Write report.csv using Pandas if available, otherwise standard CSV module
        csv_path = output_dir / "report.csv"
        if PANDAS_OK:
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
        else:
            import csv
            if self.results:
                keys = self.results[0].keys()
                try:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        dict_writer = csv.DictWriter(f, fieldnames=keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(self.results)
                except Exception as e:
                    print(f"ERROR: Failed to write CSV report: {e}")

        # Print final counts summary
        pass_count = sum(1 for r in self.results if r["overall_status"] == "PASS")
        review_count = sum(1 for r in self.results if r["overall_status"] == "REVIEW")
        reject_count = sum(1 for r in self.results if r["overall_status"] == "REJECT")

        print(f"\nDone. Report saved to: {csv_path}")
        print(f"Final QC Summary counts:")
        print(f"  PASS:   {pass_count}")
        print(f"  REVIEW: {review_count}")
        print(f"  REJECT: {reject_count}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Dataset Quality Control (QC) Tool. "
                    "Filters and sorts candidate images based on sharpness, exposure, "
                    "face identity match, pose ratios, hand configuration, and duplicate detection."
    )
    parser.add_argument(
        "--anchors", required=True, type=Path,
        help="Path to the directory containing reference character/identity face images."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the directory containing candidate images to run through QC."
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to the output directory where results (pass/review/reject subdirectories and report.csv) will be saved."
    )
    args = parser.parse_args()

    # Validate directory existence
    if not args.anchors.exists():
        sys.exit(f"ERROR: Anchors directory not found: {args.anchors}")
    if not args.input.exists():
        sys.exit(f"ERROR: Input directory not found: {args.input}")

    thresholds = Thresholds()
    pipeline = LoRAQCPipeline(thresholds, args.anchors)
    pipeline.process(args.input, args.output)


if __name__ == "__main__":
    main()
