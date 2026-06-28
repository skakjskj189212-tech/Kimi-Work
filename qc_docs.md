# LoRA Dataset QC Tool Documentation & Tuning Checklist

This guide outlines the setup, execution, visual inspection workflow, and threshold tuning strategy for the LoRA Dataset Quality Control (QC) pipeline.

---

## 1. Setup & Folder Structure

### Installation
Activate your virtual environment and install the required dependencies (optional heavy models degrade gracefully if not installed):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install opencv-python pillow numpy pandas imagehash tqdm torch torchvision facenet-pytorch mediapipe
```
> [!NOTE]
> `facenet-pytorch` and `mediapipe` will download their pre-trained weights on the first run. Subsequent executions are fully offline.

### Directory Layout
Before running the script, arrange your files as follows:
```text
project/
├── lora_qc.py
├── docs.md              # This documentation file
├── anchors/             # 3–5 clean reference face crops of your character
├── generated/           # 20–30 pilot images to be filtered
└── output/              # Created automatically by the script
    ├── pass/            # High-quality candidate images ready for training
    ├── review/          # Borderline/uncertain images (manually inspect first!)
    ├── reject/          # Blurry, duplicated, or badly clipped images
    └── report.csv       # Unified CSV report containing metrics, notes, and reasons
```

### Run Command
Execute the Quality Control pipeline via terminal:
```bash
python lora_qc.py --anchors ./anchors --input ./generated --output ./output
```

---

## 2. Post-Run Visual Inspection Checklist

Use this 8-step checklist when inspecting the first 20–30 pilot images to verify classification accuracy.

### 1. Review the `review/` Folder First
Do not start with `reject/`. Open `output/review/` first. These borderline cases will show whether your thresholds are too strict or too loose. Check each image for:
* [ ] **Identity consistency**: Does the face match the anchors?
* [ ] **Face detail**: Is the face clean, clear, and unwarped?
* [ ] **Anatomy**: Are the hands, fingers, and body proportions natural?
* [ ] **Stylistic blur**: Is the depth of field / bokeh level acceptable?
* [ ] **Exposure**: Does the lighting match the desired aesthetic?
* [ ] **Uniqueness**: Is the image a duplicate of a cleaner image in `pass/`?

### 2. Confirm the Classification is Reasonable
For every image in `output/review/`, evaluate:
* *Is this image actually good enough to train on?*
* *Did the script flag it for the right reason (check the `reason` column in `report.csv`)?*
* *Is the failure minor and salvageable, or truly problematic?*
* **Decision**: If the image is good, note which metric triggered the review so you can relax that threshold. If it is bad, keep or tighten the threshold.

### 3. Check the `reject/` Folder
Inspect `output/reject/` for **false rejects** to detect over-aggressive thresholds. Watch for:
* [ ] Close-up portraits with tight crops.
* [ ] Stylized or soft-focus artistic images.
* [ ] Dark clothing, dark hair, or moody lighting (punished by exposure checks).
* [ ] Full-body shots with tiny, distant hands.
* [ ] Profile angles that weakened facial similarity scores.
* [ ] Mildly altered variants from the same generation seed.

### 4. Compare Borderline Identity Cases against Anchors
Compare candidates visually against your `anchors/` set. Look for:
* [ ] Facial structure consistency (eyes, nose, jawline stability).
* [ ] Same overall character feel.
* [ ] Whether costume, lighting, or style changes explain the score drop.
* **Decision**: If the character is correct but the score is low, your identity thresholds are too strict.

### 5. Inspect Duplicate Clusters
If multiple images from the same prompt or seed are clustered together:
* Keep only the best one.
* Discard nearly identical variants.
* Prefer the image with the cleanest face, hands, and overall composition.
* **Decision**: If the script is flagging acceptable variations as duplicates, loosen the near-duplicate distance.

### 6. Adjust for Pose & Hand Noise
* For portrait-heavy datasets, treat pose and hand warnings as **review triggers**, not automatic rejections.
* Rely on these checks heavily for full-body shots, but discount them for tight crops and portraits where limbs/extremities are naturally cut off.

### 7. Update Thresholds in Order
To avoid changing too many variables at once, tune thresholds one at a time in the following order:
1. **Identity**
2. **Blur**
3. **Exposure**
4. **Duplicates**
5. **Pose**
6. **Hands**

### 8. Finalize the LoRA Training Set
Only images in `output/pass/` (or manually approved from `output/review/`) should enter your final training folder. Ensure they are:
* [ ] Clearly correct for identity.
* [ ] Visually clean and sharp.
* [ ] Not near-duplicates.
* [ ] Free of major hand or anatomy distortions.

---

## 3. How to Analyze `report.csv`

Open `output/report.csv` and sort by the following columns to quickly identify outliers:
* `overall_status`: Inspect `REJECT` and `REVIEW` blocks.
* `reason`: Read the consolidated summary of which metrics failed.
* `identity_score`: Identify where face matching is borderline (around `0.5` to `0.7`).
* `blur_score`: Locate soft-focus images (low laplacian variance) or over-sharpened images.
* `exposure_black_pct` / `exposure_white_pct`: Identify crushed shadows or blown highlights.
* `duplicate_note`: Find clusters of identical/near-identical seeds.

---

## 4. Threshold Tuning Reference

| Scorer | Metric / CSV Column | Tuning Advice |
| :--- | :--- | :--- |
| **Identity** | `identity_score` | • If good images get `REVIEW` too often, lower `IDENTITY_REVIEW` (e.g. to `0.35`).<br>• If incorrect faces pass, raise `IDENTITY_PASS`.<br>• For costume changes, side profiles, or heavy styling, keep a wider `REVIEW` band. |
| **Blur** | `blur_score` | • Tune blur based on dataset resolution. For `1024x1024` images, `120` is a safe sharpness floor. For `512x512`, try `80`.<br>• If soft-focus, bokeh, or hand-painted art is rejected, lower the strictness floor. |
| **Exposure** | `exposure_black_pct`<br>`exposure_white_pct` | • If moody lighting or dark clothing is punished, raise the black clipping tolerance (`CLIP_BLACK_REJECT`).<br>• Treat exposure as style-aware, not absolute. |
| **Duplicates**| `duplicate_note` | • If same-seed variants are flagged too aggressively, loosen `DUP_REVIEW_DIST`.<br>• If identical crops/images leak through, tighten `DUP_REJECT_DIST` (e.g., to `0` or `1`). |
| **Pose** | `pose_shoulder_hip_ratio`<br>`pose_head_body_ratio` | • Use pose checks mainly for half/full-body shots.<br>• For portraits, partial poses automatically pass. If stylized anatomy causes false warnings, widen shoulder/hip and head/body ranges. |
| **Hands** | `hand_conf`<br>`hand_note` | • Keep hand checks sensitive but not fatal. A low-confidence or partially visible hand should trigger `REVIEW` instead of `REJECT` unless the failure is visually egregious. |
