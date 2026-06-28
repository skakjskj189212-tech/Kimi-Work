# Pinterest Reference Collection Workflow — Step-by-Step

> **Goal:** Go from zero to ComfyUI-ready anchor images in under 45 minutes.  
> **Character:** Nyra (Mixed European-American, 24, attractive, athletic-slim)  
> **Output:** 5 face anchors + 20–30 categorized reference images

---

## Prerequisites

- Pinterest account (free)
- Chrome or Edge browser
- Python 3.10+ with the QC dependencies installed
- This workspace folder open in a terminal

---

## Step 1: Create Folder Structure (2 minutes)

Open terminal in your workspace and run:

```bash
# Windows (Git Bash / PowerShell)
mkdir -p archetype_anchors pinterest_raw output

# Or in Python
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['archetype_anchors', 'pinterest_raw', 'output']]"
```

Your folder tree should look like:
```
workspace/
├── archetype_anchors/     # YOU manually curate 5-10 perfect images here
├── pinterest_raw/          # Bulk downloaded images land here
├── output/                 # QC results go here
├── bulk_download.py
├── pinterest_qc.py
├── pinterest_qc_pro.py     # AI-enhanced version
├── pinterest_search_terms.md
└── character_definition.md
```

---

## Step 2: Manually Curate Archetype Anchors (10 minutes)

**This is the ONLY manual step you cannot automate.**

1. Open Pinterest in your browser: https://pinterest.com
2. Search using these **exact terms** (one at a time):
   - `attractive woman face oval high cheekbones natural makeup`
   - `woman light brown hair wavy voluminous portrait`
   - `hazel green eyes woman portrait soft light`
   - `all american beauty girl next door portrait`
   - `woman warm beige skin tone natural beauty portrait`
3. Look for images that match **all** of these Nyra traits:
   - [ ] Heart-shaped to oval face
   - [ ] High cheekbones, defined jawline
   - [ ] Light brown to dark blonde hair
   - [ ] Hazel-green or warm brown eyes
   - [ ] Warm beige skin with light olive undertone
   - [ ] Athletic-slim but not underweight
   - [ ] Modern, approachable, photorealistic (not filtered/illustrated)
4. Save **5–10 images** to your `archetype_anchors/` folder
   - Right-click → "Save image as" → navigate to `workspace/archetype_anchors/`
   - Name them: `archetype_01.jpg`, `archetype_02.jpg`, etc.

> **Critical:** These images are your "ground truth." The QC will compare every downloaded image against these. If your archetypes are wrong, everything downstream is wrong. Spend the full 10 minutes.

---

## Step 3: Search Pinterest & Export URLs (10 minutes)

### Method A: Board + RSS Export (Fastest)

1. In Pinterest, create a **secret board** called `Nyra Raw Refs`
2. Search each term from `pinterest_search_terms.md` and save 15–20 pins to the board
   - Save more than you need — the QC will filter them down
   - Aim for 60–80 total pins across all categories
3. Once the board is full, get the RSS feed URL:
   ```
   https://www.pinterest.com/<YOUR_USERNAME>/nyra-raw-refs.rss
   ```
4. Open that URL in a new tab. You'll see raw XML with `<enclosure url="...">` tags containing image URLs.
5. Copy all the image URLs to a text file:
   - Press `Ctrl+U` to view page source
   - Press `Ctrl+F`, search for `.jpg` or `pinimg.com`
   - Copy all URLs to `pinterest_urls.txt` in your workspace

### Method B: Browser Console Export (More Control)

1. Open your Pinterest board with all saved pins
2. Scroll down to load all pins (Pinterest lazy-loads)
3. Press `F12` → Console tab
4. Paste and run this JavaScript:
   ```javascript
   // Extract all visible image URLs from Pinterest board
   const images = [...document.querySelectorAll('img[src*="pinimg.com"]')]
     .map(img => img.src)
     .filter(src => src.includes('736x') || src.includes('originals'))
     .map(src => src.replace(/\d+x\d+/, 'originals'))  // get full resolution
     .filter((v, i, a) => a.indexOf(v) === i);  // deduplicate
   console.log(images.join('\n'));
   copy(images.join('\n'));  // copies to clipboard
   ```
5. Paste the URLs into `pinterest_urls.txt` in your workspace

> **Pro tip:** Replace `736x` with `originals` in URLs to get the highest resolution images.

---

## Step 4: Bulk Download (2 minutes)

```bash
# Run the bulk downloader
python bulk_download.py --urls pinterest_urls.txt --output ./pinterest_raw --threads 8
```

Expected output:
```
Found 72 URLs to download
Output directory: C:\Users\...\workspace\pinterest_raw
Parallel threads: 8
--------------------------------------------------
[OK] ./pinterest_raw/nyra_ref_001.jpg
[OK] ./pinterest_raw/nyra_ref_002.jpg
...
Download complete: 68 success, 4 failed out of 72 total
```

**If bulk download fails** (Pinterest blocks some URLs):
- The script will report which URLs failed
- Manually download the failed ones, or skip them — 60+ images is plenty

---

## Step 5: Run AI-Enhanced QC (3–5 minutes)

### Option A: Basic QC (runs immediately, no new installs)
```bash
python pinterest_qc.py --archetype ./archetype_anchors --input ./pinterest_raw --output ./output
```

### Option B: AI-Enhanced Production QC (requires install, see below)
```bash
# Install AI dependencies (one-time, ~2GB download)
pip install transformers timm insightface

# Run the production QC
python pinterest_qc_pro.py --config qc_config.yaml --archetype ./archetype_anchors --input ./pinterest_raw --output ./output
```

The production QC uses:
- **CLIP** (transformers) for semantic similarity: "does this image match the Nyra description?"
- **InsightFace** (buffalo_l) for face recognition + face quality scoring
- **Ensemble scoring** combining 6 AI + CV scorers into a 0–100 production score
- **YAML config** for all thresholds — edit without touching code

---

## Step 6: Review Results & Pick Anchors (10 minutes)

After QC runs, open the output:

```
output/
├── face_anchors/          ← Your ComfyUI ReferenceLatent images
├── hair_refs/
├── outfit_refs/
├── pose_refs/
├── full_body_refs/
├── approved_all/
├── review/                ← Check these manually
├── reject/                ← Already filtered out
├── report.csv             ← Open in Excel / VS Code
└── report.json            ← (production QC only) Per-image full metadata
```

### What to do:

1. **Open `report.csv`** — sort by `category` then `overall_score` descending
2. **Check `face_anchors/`** — these should be clean close-up portraits
   - Pick the **top 5** by overall_score for your ComfyUI `ReferenceLatent` node
   - Copy them to a new folder: `comfyui_anchors/`
3. **Check `review/`** — some might be good but borderline
   - Open each image. If it looks good, manually move it to `approved_all/`
4. **Ignore `reject/`** — these are already filtered out
5. **Check `hair_refs/`, `outfit_refs/`, `full_body_refs/`** — these are your secondary references for prompt writing

### Quick validation checklist for face anchors:
- [ ] Face is clearly visible and well-lit
- [ ] Expression is neutral to slightly smiling
- [ ] Hair is visible and matches Nyra's description
- [ ] No heavy filters or extreme editing
- [ ] Different angles: front, 3/4, and one side profile

---

## Step 7: Copy Anchors to ComfyUI (1 minute)

```bash
# Create ComfyUI anchor folder
mkdir -p comfyui_anchors

# Copy top 5 face anchors (adjust filenames as needed)
cp output/face_anchors/nyra_ref_003.jpg comfyui_anchors/anchor_01_front.jpg
cp output/face_anchors/nyra_ref_007.jpg comfyui_anchors/anchor_02_threequarter.jpg
cp output/face_anchors/nyra_ref_012.jpg comfyui_anchors/anchor_03_profile.jpg
cp output/face_anchors/nyra_ref_015.jpg comfyui_anchors/anchor_04_smile.jpg
cp output/face_anchors/nyra_ref_021.jpg comfyui_anchors/anchor_05_closeup.jpg
```

These 5 images are now your **ReferenceLatent anchor set** for ComfyUI.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Pinterest images are low-res | Search for `originals` in URLs, or use the `.rss` method |
| Bulk download fails with 403 | Pinterest blocks some CDN URLs. Try fewer threads (`--threads 3`) or add `--delay 1.0` |
| QC says "facenet_unavailable" | Run `pip install facenet-pytorch torch` |
| No face detected in archetype | Your archetype image doesn't have a visible face. Replace it |
| All images go to `reject/` | Your archetype is too strict or doesn't match the downloaded images. Adjust thresholds in `qc_config.yaml` |
| AI QC says "transformers not installed" | Run `pip install transformers timm` (~1.5GB download) |
| InsightFace model not found | Run once: `python -c "import insightface; insightface.app.ImageAnalysisApp(name='buffalo_l')"` to auto-download |

---

## Next Step After This Workflow

Once you have your 5 ComfyUI anchors, the next action is:

> **Generate the pilot batch** — 25 ComfyUI prompts (5 anchors + 20 variable) using FLUX.2-dev + ReferenceLatent

I'll build those prompts for you when you're ready.

---

*File: pinterest_workflow_guide.md | Step-by-step guide to go from Pinterest search to ComfyUI anchors*
