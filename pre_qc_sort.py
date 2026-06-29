#!/usr/bin/env python3
"""
Pre-QC Image Sorter
===================
Fast, lightweight pre-filter before running the full pinterest_qc_pro.py pipeline.

Scans all images in ./downloaded_images/ subfolders and applies:
  1. Minimum file size      - reject corrupt/tiny files
  2. Minimum resolution     - reject small images
  3. Sharpness (blur check) - reject blurry images
  4. Exposure check         - reject over/under-exposed images
  5. Face detection         - reject images with no face
  6. Deduplication          - reject near-identical images (perceptual hash)

Good images are COPIED into ./pinterest_raw/ (flat, ready for QC scripts).
Rejected images are logged but NOT moved.

Usage:
    python pre_qc_sort.py                          # auto-scan ./downloaded_images/
    python pre_qc_sort.py --input ./my_downloads   # custom input folder
    python pre_qc_sort.py --output ./pinterest_raw # custom output folder
    python pre_qc_sort.py --dry-run                # preview without copying
    python pre_qc_sort.py --min-res 768            # stricter resolution (default 512)
    python pre_qc_sort.py --blur-thresh 100        # stricter blur (default 80)
    python pre_qc_sort.py --no-face-check          # skip face detection

Dependencies (all lightweight, no GPU needed):
    pip install opencv-python pillow numpy imagehash tqdm
"""

import os
import sys
import shutil
import argparse
import csv
from pathlib import Path

# ---- Core deps ---------------------------------------------------------------
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    print("[ERROR] Missing core deps. Run: pip install opencv-python pillow numpy")
    sys.exit(1)

try:
    import imagehash
    HASH_OK = True
except ImportError:
    HASH_OK = False
    print("[WARN] imagehash not found. Deduplication disabled. Run: pip install imagehash")

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False
    def tqdm(it, *a, **k):
        return it

# ---- Config defaults ---------------------------------------------------------
DEFAULTS = dict(
    min_file_kb  = 20,     # reject files smaller than 20 KB
    min_res      = 512,    # reject if width OR height < 512px
    blur_thresh  = 80.0,   # Laplacian variance; below = blurry
    clip_black   = 25.0,   # % of pixels near-black before reject
    clip_white   = 20.0,   # % of pixels near-white before reject
    hash_dist    = 6,      # perceptual hash distance; below = duplicate
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# ---- OpenCV face detector (Haar cascade, zero extra downloads) ---------------
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(cascade_path)
    if clf.empty():
        print("[WARN] Face cascade not found. Face check disabled.")
        return None
    return clf


# ---- Per-image checks --------------------------------------------------------

def check_file_size(path, min_kb):
    kb = path.stat().st_size / 1024
    if kb < min_kb:
        return False, "file too small ({:.1f} KB < {} KB)".format(kb, min_kb)
    return True, ""


def check_resolution(img_cv, min_res):
    h, w = img_cv.shape[:2]
    if w < min_res or h < min_res:
        return False, "too small ({}x{} < {}px)".format(w, h, min_res)
    return True, "{}x{}".format(w, h)


def check_blur(img_cv, thresh):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < thresh:
        return False, "blurry (score {:.1f} < {})".format(lap_var, thresh)
    return True, "sharpness={:.1f}".format(lap_var)


def check_exposure(img_cv, black_thresh, white_thresh):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    total = gray.size
    pct_black = float((gray < 15).sum()) / total * 100
    pct_white = float((gray > 240).sum()) / total * 100
    if pct_black > black_thresh:
        return False, "underexposed ({:.1f}% black > {}%)".format(pct_black, black_thresh)
    if pct_white > white_thresh:
        return False, "overexposed ({:.1f}% white > {}%)".format(pct_white, white_thresh)
    return True, "exposure OK"


def check_face(img_cv, detector):
    if detector is None:
        return True, "face_check=skipped"
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return False, "no face detected"
    return True, "faces={}".format(len(faces))


def compute_hash(pil_img):
    if not HASH_OK:
        return None
    return imagehash.phash(pil_img, hash_size=16)


# ---- Collect all images ------------------------------------------------------

def collect_images(input_root):
    images = []
    for p in sorted(input_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


# ---- Main pipeline -----------------------------------------------------------

def run(args):
    input_root = Path(args.input)
    output_dir = Path(args.output)
    dry_run    = args.dry_run
    face_check = not args.no_face_check

    if not input_root.exists():
        print("[ERROR] Input folder not found: {}".format(input_root))
        sys.exit(1)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("PRE-QC IMAGE SORTER")
    print("="*60)
    print("  Input  : {}".format(input_root.absolute()))
    print("  Output : {}".format(output_dir.absolute()))
    print("  Checks : res>={}px | blur>={} | face={} | dedup={}".format(
        args.min_res, args.blur_thresh,
        "yes" if face_check else "no",
        "yes" if HASH_OK else "no"
    ))
    print("  Mode   : {}".format(
        "DRY RUN - no files will be copied" if dry_run else "LIVE"
    ))
    print("="*60 + "\n")

    all_images = collect_images(input_root)
    if not all_images:
        print("[ERROR] No images found in input folder.")
        sys.exit(1)
    print("[INFO] Found {} images to process.\n".format(len(all_images)))

    detector = load_face_detector() if face_check else None
    seen_hashes = {}  # hash -> first file path

    stats = dict(
        total=0, passed=0,
        failed_size=0, failed_res=0, failed_blur=0,
        failed_exposure=0, failed_face=0, failed_dup=0,
    )
    report_rows = []

    for img_path in tqdm(all_images, desc="Sorting", unit="img"):
        stats["total"] += 1
        row = {
            "file": img_path.name,
            "source_folder": img_path.parent.name,
            "result": "pass",
            "reason": "",
        }

        # 1. File size
        ok, msg = check_file_size(img_path, args.min_file_kb)
        if not ok:
            stats["failed_size"] += 1
            row.update(result="reject", reason=msg)
            report_rows.append(row)
            print("  [REJECT] {} | {}".format(img_path.name, msg))
            continue

        # 2. Load image
        try:
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                raise ValueError("cv2.imread returned None (corrupt?)")
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            stats["failed_size"] += 1
            row.update(result="reject", reason="corrupt: {}".format(e))
            report_rows.append(row)
            print("  [REJECT] {} | corrupt: {}".format(img_path.name, e))
            continue

        # 3. Resolution
        ok, msg = check_resolution(img_cv, args.min_res)
        if not ok:
            stats["failed_res"] += 1
            row.update(result="reject", reason=msg)
            report_rows.append(row)
            print("  [REJECT] {} | {}".format(img_path.name, msg))
            continue

        # 4. Blur / sharpness
        ok, detail = check_blur(img_cv, args.blur_thresh)
        if not ok:
            stats["failed_blur"] += 1
            row.update(result="reject", reason=detail)
            report_rows.append(row)
            print("  [REJECT] {} | {}".format(img_path.name, detail))
            continue

        # 5. Exposure
        ok, detail = check_exposure(img_cv, args.clip_black, args.clip_white)
        if not ok:
            stats["failed_exposure"] += 1
            row.update(result="reject", reason=detail)
            report_rows.append(row)
            print("  [REJECT] {} | {}".format(img_path.name, detail))
            continue

        # 6. Face detection
        if face_check:
            ok, detail = check_face(img_cv, detector)
            if not ok:
                stats["failed_face"] += 1
                row.update(result="reject", reason=detail)
                report_rows.append(row)
                print("  [REJECT] {} | {}".format(img_path.name, detail))
                continue

        # 7. Deduplication
        if HASH_OK:
            h = compute_hash(pil_img)
            dup_found = None
            for existing_hash, existing_path in seen_hashes.items():
                if abs(h - existing_hash) <= args.hash_dist:
                    dup_found = existing_path
                    break
            if dup_found:
                stats["failed_dup"] += 1
                row.update(result="reject", reason="duplicate of {}".format(dup_found.name))
                report_rows.append(row)
                print("  [DEDUP]  {} -> dup of {}".format(img_path.name, dup_found.name))
                continue
            seen_hashes[h] = img_path

        # ---- PASSED all checks ----
        stats["passed"] += 1
        row["result"] = "pass"
        report_rows.append(row)

        # Copy to output (handle name collisions)
        dest = output_dir / img_path.name
        if dest.exists():
            stem   = img_path.stem
            suffix = img_path.suffix
            tag    = img_path.parent.name[:10].replace(" ", "_")
            dest   = output_dir / "{}_{}{}".format(stem, tag, suffix)

        if not dry_run:
            shutil.copy2(img_path, dest)
            print("  [PASS]   {}  ->  {}".format(img_path.name, dest.name))
        else:
            print("  [PASS]   {}  (would copy)".format(img_path.name))

    # ---- Write CSV report ----------------------------------------------------
    if not dry_run:
        report_path = output_dir / "pre_qc_report.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "source_folder", "result", "reason"]
            )
            writer.writeheader()
            writer.writerows(report_rows)
        print("\n  [INFO] Report saved to: {}".format(report_path))

    # ---- Summary -------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("  Total images scanned   : {}".format(stats["total"]))
    print("  [PASS] Copied to QC    : {}".format(stats["passed"]))
    print("  --- Reject breakdown ---")
    print("  File too small/corrupt : {}".format(stats["failed_size"]))
    print("  Too low resolution     : {}".format(stats["failed_res"]))
    print("  Too blurry             : {}".format(stats["failed_blur"]))
    print("  Bad exposure           : {}".format(stats["failed_exposure"]))
    print("  No face detected       : {}".format(stats["failed_face"]))
    print("  Duplicate              : {}".format(stats["failed_dup"]))
    total_reject = stats["total"] - stats["passed"]
    print("  --- Total rejected     : {}".format(total_reject))
    print("\n  Output: {}".format(output_dir.absolute()))
    if dry_run:
        print("  [DRY RUN] No files were copied.")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-QC image sorter: filter good images into pinterest_raw/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",         default="./downloaded_images",
                        help="Root folder of downloaded images (default: ./downloaded_images)")
    parser.add_argument("--output",        default="./pinterest_raw",
                        help="Output folder for passing images (default: ./pinterest_raw)")
    parser.add_argument("--min-res",       type=int,   default=DEFAULTS["min_res"],
                        help="Min width/height in pixels (default: {})".format(DEFAULTS["min_res"]))
    parser.add_argument("--blur-thresh",   type=float, default=DEFAULTS["blur_thresh"],
                        help="Min Laplacian variance for sharpness (default: {})".format(DEFAULTS["blur_thresh"]))
    parser.add_argument("--clip-black",    type=float, default=DEFAULTS["clip_black"],
                        help="Max pct near-black pixels (default: {})".format(DEFAULTS["clip_black"]))
    parser.add_argument("--clip-white",    type=float, default=DEFAULTS["clip_white"],
                        help="Max pct near-white pixels (default: {})".format(DEFAULTS["clip_white"]))
    parser.add_argument("--hash-dist",     type=int,   default=DEFAULTS["hash_dist"],
                        help="Perceptual hash distance for dedup (default: {})".format(DEFAULTS["hash_dist"]))
    parser.add_argument("--min-file-kb",   type=float, default=DEFAULTS["min_file_kb"],
                        help="Min file size in KB (default: {} KB)".format(DEFAULTS["min_file_kb"]))
    parser.add_argument("--no-face-check", action="store_true",
                        help="Skip face detection (faster)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Preview results without copying any files")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
