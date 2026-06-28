# Girl Character LoRA — Complete Project Context

> **Consolidated from:** QC Pipeline files (`D:\QC CHECK`), Perplexity Chat 1 (Gemini Deep Research Prompt), and Perplexity Chat 2 (Project Direction & Details)
> **Date:** June 28, 2026
> **User:** Saransh Kumar, Bengaluru, India — AI engineer / ML practitioner / cybersecurity enthusiast
> **Hardware:** RTX PRO 6000 96GB VRAM
> **Primary Tool:** ComfyUI (fully local) + FLUX.2-dev full precision

---

## 0. User Background & Working Style

- **Name:** Saransh Kumar
- **Location:** Bengaluru, India
- **Role:** AI engineer / ML practitioner / cybersecurity enthusiast
- **Technical stack:** Python, Bash, CUDA, PyTorch, GPU workflows, AI model hosting, quantization, automation, debugging
- **Tools used:** Kaggle notebooks, NVIDIA NIM, Lightning AI, Hugging Face, n8n, Apify, Playwright, Kali Linux, VS Code
- **Working style:** Highly iterative — prototype, test, inspect failures, tune, repeat
- **Preferences:** Actionable, concise, engineering-focused answers tuned for real implementation, not theory
- **Project philosophy:** Practical, local, offline-friendly, pilot-driven before scaling

---

## 1. Project Overview

This is a **local QC and dataset engineering project** for a **female character LoRA**. The goal is to build a production-ready workflow that:

1. **Defines** a visually specific, attractive female character using reference direction (Pinterest-style moodboard approach).
2. **Generates** a controlled dataset of that character using FLUX.2-dev in ComfyUI (fully local).
3. **Scores** every generated image automatically for identity, body, anatomy, quality, and duplication.
4. **Filters** images into `pass`, `review`, and `reject` buckets via a local Python QC pipeline.
5. **Trains** a LoRA only on the approved images, producing a model that can generate:
   - Realistic full-body portraits
   - Extreme close-up portraits
   - Consistent face identity, hair, eyes, skin tone, face shape, and body proportions

The entire workflow is **local, offline-friendly, and practical**.

---

## 2. Core Objective

Ensure the character remains **visually consistent** across many images without overfitting the model to:
- The same pose
- The same outfit
- The same background
- The same lighting

The QC pipeline answers one question repeatedly: **"Is this image good enough to train on?"**

If the answer is not clearly **yes**, it goes to **review** or **reject**.

---

## 3. Character Design Direction

The character is **AI-generated**, not a real person. Her visual identity must be deliberately defined before generation begins.

### Reference Direction (Pinterest-style moodboard)
The reference collection should be organized by:

- **Face direction** — specific face shape, eye shape, nose, jawline, expression style
- **Hair direction** — color, style, length, texture, common hairstyles
- **Outfit direction** — variety of clothing types (see Clothing Section below)
- **Pose direction** — standing, sitting, walking, looking at camera, profile angles
- **Lighting direction** — soft, dramatic, studio, golden hour, moody
- **Mood / style direction** — aesthetic feel (elegant, casual, edgy, etc.)

The goal is to make the character **visually attractive, specific, and controlled** — not vague or generic.

### Consistency Requirements
The trained LoRA must preserve these across all generated outputs:

| Consistency Aspect | Requirement |
| :--- | :--- |
| **Face identity** | Same person across close-ups, portraits, and full-body shots |
| **Hair** | Consistent color, texture, and style |
| **Eyes** | Same eye shape, color, and expression style |
| **Skin tone** | Consistent across all lighting conditions |
| **Face shape** | Stable jawline, cheekbone, and nose structure |
| **Body shape** | Consistent proportions, waist-to-hip ratio, silhouette |
| **Body proportions** | Realistic anatomy, no drift in limb length or torso shape |
| **Realism** | Maximum realism, photorealistic quality |

---

## 4. Dataset Plan

### Target Size
- **Minimum:** ~80 images
- **Ideal:** 120–200 images
- **Maximum:** ~300 images (to avoid overfitting)

### Image Type Distribution
The dataset should be **portrait-biased** with some full-body coverage:

| Shot Type | Proportion | Purpose |
| :--- | :--- | :--- |
| **Close-ups** | ~20% | Strong identity learning for face features |
| **Portraits** | ~30% | Face + upper body, identity stability |
| **Half-body / bust** | ~30% | Outfit + pose + body shape learning |
| **Full-body** | ~15% | Silhouette, proportions, full outfit |
| **Contextual / environmental** | ~5% | Scene context, lighting variety |

> **Why portrait-biased?** The face is the primary identity carrier. Full-body shots where the face is too small are less useful for identity learning.

### Resolution & Aspect Ratio
- **Recommended resolution:** 1024×1024 (square) or 1024×1280 (portrait)
- **Aspect ratio:** Primarily square or portrait-oriented
- **Rationale:** FLUX.2-dev is trained on 1024px images. Square is safest for identity learning. Portrait (3:4 or 2:3) works well for full-body if the face remains large enough.

### Outfit Strategy
The dataset must include **varied outfits** so the model does not overfit to a single clothing style.

**Clothing types to include:**
- Casual tops and jeans
- Dresses (varied styles)
- Fitted/form-fitting outfits (anatomical baseline)
- Stylish streetwear
- Minimal / revealing but controlled attire (for body baseline)
- Practical and fashionable looks
- Some minimal, form-fitting attire as **anatomical baseline** — helps the model understand body structure

**Important rule:** Vary outfits, backgrounds, and lighting in every image. Do not let the model memorize a single look.

---

## 5. Generation Workflow (ComfyUI + FLUX.2-dev)

### Hardware Context
- **GPU:** RTX PRO 6000 96GB VRAM
- **Model:** FLUX.2-dev full precision (`flux2-dev.safetensors`)
- **Environment:** ComfyUI, fully local
- **VRAM budget:** 96GB allows full precision without offloading — prioritize quality over speed

### Model Details (from Hugging Face)
- **Architecture:** 32B parameter rectified-flow transformer
- **Purpose:** Image generation and editing
- **License:** Non-commercial (FLUX.2-dev license)
- **Base setup:** Model + required text encoders + VAE (ComfyUI stack)

### Generation Method
- **Primary:** Text-to-image (no reference images during generation)
- **Goal:** Max realism
- **Strategy:** Controlled prompts with varied dynamic attributes

### Prompt Structure
Each generation prompt should describe **dynamic, visible features**:

```
[subject trigger], [pose description], [outfit description], [lighting],
[environment], [camera style], [expression], [composition details]
```

**What to include:**
- Pose (standing, sitting, walking, looking over shoulder)
- Outfit (specific clothing item and style)
- Environment (indoor, outdoor, studio, city, nature)
- Lighting (soft, dramatic, golden hour, studio lighting)
- Camera style (DSLR, shallow depth of field, 85mm lens, etc.)
- Expression (smiling, neutral, confident, playful)
- Framing / composition

**What to AVOID in prompts:**
- Permanent identity traits (eye color, hair color, face shape) — these should be carried by the **trigger word** and the **dataset**, not the prompt text
- Overloading captions with identity descriptors — prevents the trigger word from learning the identity properly

---

## 6. QC Pipeline (Local Python — `lora_qc.py`)

### Purpose
Automatically score every generated image and sort into `pass`, `review`, or `reject` before training.

### Output Structure
```
output/
├── pass/          # Approved for training
├── review/        # Borderline — manual inspection required
├── reject/        # Obvious failures
└── report.csv     # Scores, reasons, and status for every image
```

### QC Gates (in order of priority)

#### 1. Identity Gate
- Face detection (MTCNN / FaceNet)
- Face embeddings
- Compare candidate faces to anchor/reference faces
- **Low matches → REVIEW** (not reject, to avoid false rejects on profile shots or extreme angles)
- **Multiple faces → REVIEW**

#### 2. Body / Proportion Gate
- Pose keypoint detection (MediaPipe)
- Shoulder-to-hip ratio check
- Head-to-body ratio check
- **Portrait / close-up shots automatically pass** (keypoints not visible = partial pose = OK)
- **Full-body shots** are checked more strictly

#### 3. Hand / Anatomy Gate
- Hand detection (MediaPipe Hands)
- Check for malformed hands, missing fingers, low confidence
- **Low confidence → REVIEW** (not reject, unless visually egregious)
- **Clustered landmarks (anomaly) → REVIEW**
- **Small/distant hands → soft pass** (avoid false positives)

#### 4. Quality Gate
- **Blur:** Laplacian variance (sharpness score)
  - 1024px images: floor ~120
  - 512px images: floor ~80
- **Exposure:** Black/white clipping percentage
  - Avoid false rejects on dark hair, dark clothing, moody lighting

#### 5. Deduplication Gate
- Perceptual hash (phash) + difference hash (dhash)
- Cluster near-identical images
- Keep only the best representative from each cluster

#### 6. Consistency Gates (Auto-Evaluation)
- **Face similarity** across all images (embedding distance)
- **Hair consistency** (color/style embedding or visual comparison)
- **Eye consistency** (same shape, color, alignment)
- **Skin tone consistency** (color histogram or LAB space comparison)
- **Face shape consistency** (landmark stability)
- **Body shape / limb proportion consistency** (pose landmark variance)
- **Overall identity drift detection**

### Threshold Philosophy
- **Hard reject only obvious failures** (blur < 30, extreme exposure, exact duplicates, wrong identity)
- **Borderline → REVIEW** (soften harsh rules to avoid false rejects)
- **Thresholds are centralized and easy to edit** (in `Thresholds` dataclass)
- **Tune through pilot, not guess on day one**

### Threshold Tuning Order
1. **Identity** — most fundamental
2. **Blur** — technical quality baseline
3. **Exposure** — style-aware, not absolute
4. **Duplicates** — prevent overfitting
5. **Pose** — body proportion stability
6. **Hands** — anatomical quality

### Pilot Review Workflow
1. **Inspect `review/` first** — shows if thresholds are too strict or too loose
2. **Check `reject/` for false rejects** — detect over-aggressive thresholds
3. **Sort `report.csv` by:** `overall_status`, `reason`, `identity_score`, `blur_score`
4. **Compare CSV with actual images** — visual cross-reference
5. **Tune one threshold at a time** — avoid changing multiple variables
6. **Re-run pilot** — iterate until stable

### Script Hardening (Already Discussed / Implemented)
- Duplicate-chain handling corrected
- Blur and exposure false rejects reduced
- Portrait / close-up pose handling softened
- Small / distant hand handling improved
- Multi-face detection added
- Low identity matches → REVIEW (not REJECT)
- Graceful degradation if optional models (FaceNet, MediaPipe) are missing

### 6.1 Integrated AI Brain Scorer (Production QC)

In addition to local computer vision metrics (InsightFace, MediaPipe, etc.), `pinterest_qc_pro.py` features an integrated **AI Brain Scorer** wrapper that can run in two modes:
1. **Google Antigravity CLI (`agy`)**: Invokes the `agy` command-line tool non-interactively with `--print` to run complex character alignment queries, utilizing local model permissions and reading image paths directly.
2. **Google Gemini API**: Utilizes the official `google-genai` Python SDK to call the Gemini API (`gemini-1.5-flash` or custom models) for visual character verification.

Both options parse structured JSON outputs assessing face similarity, hair consistency, eye consistency, skin tone, and photorealism, combining them into the ensemble production score with a configurable weight (default `0.35`).

---

## 7. Captioning Strategy

Captions should describe **variable, dynamic** aspects of the image. They should **NOT** overload on permanent identity traits.

### What to Include in Captions
- Pose (standing, sitting, walking, looking over shoulder)
- Outfit (red dress, leather jacket, white tank top)
- Environment (city street, studio, beach, bedroom)
- Lighting (golden hour, softbox, dramatic side light)
- Camera style (DSLR, 85mm, shallow depth of field, film grain)
- Expression (smiling, neutral, confident)
- Framing (close-up, medium shot, full body)

### What to AVOID in Captions
- Permanent identity descriptors ("green eyes", "brown hair", "oval face") — the trigger word should carry these
- Repeated identical phrases across many images
- Overly long captions that dilute the trigger word

### Example Good Caption
```
a woman standing on a city street at golden hour, wearing a fitted black leather jacket and dark jeans, confident pose, shallow depth of field, 85mm lens, soft natural lighting, looking over shoulder, medium shot
```

### Trigger Word Strategy
- Use a **unique trigger word** (e.g., `grl` or a made-up name)
- The trigger word should be the **primary identity carrier**
- Keep it short and distinct from common English words

---

## 8. LoRA Training Plan

### Training Tool
Open to tools outside ComfyUI if better. Options:
- **Kohya_ss** (most popular, GUI-based)
- **AI-Toolkit** (simple, effective)
- **Flux Gym** (Flux-specific)
- **SimpleTuner** (script-based, flexible)

### Recommended Training Settings (Single Character LoRA)

| Parameter | Recommendation | Reasoning |
| :--- | :--- | :--- |
| **Rank / Dim** | 16–32 | Identity LoRA needs moderate rank; 16 is often enough for faces, 32 for full body |
| **Alpha** | 8–16 (half of rank, or equal) | Standard practice: alpha = rank/2 for lighter training, alpha = rank for stronger learning |
| **Learning Rate** | 1e-4 to 5e-4 | Start conservative; higher LR can overfit fast on small datasets |
| **Repeats** | 10–20 per image | Enough to learn without overfitting; adjust based on dataset size |
| **Batch Size** | 2–4 (fit on 96GB VRAM) | Larger batch = more stable; 96GB allows comfortably |
| **Steps** | 1000–3000 total | Depends on dataset size × repeats; watch for overfitting |
| **Resolution** | 1024×1024 (class training) | Match FLUX.2-dev native resolution |
| **Captioning** | Natural language, trigger word first | See Captioning Strategy above |
| **Overfitting Avoidance** | Regularization images (class images) | 10–20 generic person images to prevent overfitting to outfits/backgrounds |
| **Concept Type** | Unified character LoRA (single concept) | One trigger word = one person |

### Training Tips for Portrait + Close-Up Behavior
- Include **close-up captions** in training so the model learns face details at crop level
- Use **random cropping** during training so the model sees face crops and body crops
- Balance the dataset so the model doesn't overfit to body-only or face-only
- The LoRA should work for **portraits, extreme close-ups, AND full-body** without losing identity

---

## 9. Gemini Deep Research Prompt (For ComfyUI Workflow)

This prompt was drafted to send to Gemini Deep Research to get a **practical, production-ready ComfyUI workflow**:

---

> **Prompt for Gemini Deep Research:**
>
> I need a practical, end-to-end ComfyUI workflow for generating a single AI-generated person dataset using FLUX.2-dev full precision locally on an RTX PRO 6000 96GB VRAM machine.
>
> **Goal:** Build a workflow for creating a high-quality dataset to train a LoRA that can later generate:
> - Realistic full-body portraits
> - Extreme close-ups
> - Consistent face identity, hair, eyes, skin tone, face shape, body shape, body proportions
> - Realistic anatomy and proportions across poses
>
> The subject is AI-generated, not a real person. The final LoRA should be used primarily for portraits, but the dataset must include full-body variation and be strong enough to preserve identity across close-ups and body shots.
>
> **Please research and recommend:**
> 1. **Dataset size** — How many final images? Minimum, ideal, maximum. Face-to-body ratio bias.
> 2. **Resolution and aspect ratio** — What to stick to. Square vs portrait vs mixed.
> 3. **ComfyUI generation workflow** — Practical node structure, sampler, prompts, negative prompts, seed strategy, quality enhancement steps (upscaling, detail passes, face refinement, reroll selection).
> 4. **Consistency evaluation** — Auto-check face similarity, hair consistency, eye consistency, skin tone, face shape, body shape, identity drift. Recommend local tools/nodes/scripts.
> 5. **Training workflow** — Best training settings (rank, alpha, LR, repeats, batch, steps, captioning, overfitting avoidance, regularization).
> 6. **Portrait / close-up behavior** — How to preserve consistency across close-ups and full-body without ruining either.
>
> **Output format:** Step-by-step practical guide with reasoning. Include a complete ComfyUI generation workflow, consistency evaluation pipeline, and LoRA training recipe. Production-ready, not generic overview.
>
> **Constraints:** Fully local. ComfyUI main generation env. FLUX.2-dev full precision. One AI-generated person. Maximum realism.

---

## 10. File Inventory (Saved in Workspace)

| File | Description |
| :--- | :--- |
| `character_definition.md` | Nyra's full visual identity spec (ethnicity, face, hair, body, aesthetic) |
| `pinterest_search_terms.md` | 60+ optimized Pinterest search terms organized by category |
| `pinterest_workflow_guide.md` | Step-by-step guide: search → download → QC → ComfyUI anchors |
| `bulk_download.py` | Parallel bulk image downloader with retry & rate limiting |
| `pinterest_qc.py` | Basic QC pipeline for Pinterest reference curation |
| `pinterest_qc_pro.py` | **AI-enhanced production QC** with CLIP + InsightFace + Integrated AI Brain Scorer (agy CLI / Gemini API) + ensemble scoring |
| `qc_config.yaml` | Tunable YAML config for all QC thresholds and model weights |
| `qc_docs.md` | QC documentation & tuning checklist (from `D:\QC CHECK`) |
| `qc_lora_qc.py` | Full Python QC pipeline script for generated images (from `D:\QC CHECK`) |
| `perplexity_raw_content.md` | Raw text dump from both Perplexity chats |
| `project_master_context.md` | This file — complete consolidated project context with user background |
| `continuation_prompt.md` | Paste-ready prompt for future Kimi sessions |

---

## 11. Next Steps / What to Continue

### Immediate Next Actions
1. **Run Pinterest workflow** — Search, bulk download, run AI QC (`pinterest_qc_pro.py`), pick top 5 face anchors for ComfyUI.
2. **Generate anchor images in ComfyUI** — Use FLUX.2-dev + ReferenceLatent with the 5 curated anchors to generate pristine character anchors.
3. **Design generation prompts** — Build 25 ComfyUI-ready prompts (5 anchors + 20 variable pilot images).
4. **Run pilot batch** — Generate 20–30 images using the ComfyUI workflow.
5. **Run QC pipeline** — Process pilot images through `lora_qc.py`.
6. **Inspect and tune** — Review results, adjust thresholds, re-run.
7. **Scale up** — Once pilot is stable, generate full dataset (120–200 images).
8. **Caption and train** — Apply captions, train LoRA with recommended settings.
9. **Evaluate LoRA** — Test for portrait, close-up, and full-body consistency.

### Open Questions for Next Session
- ~~Which specific trigger word to use for the character?~~ → **Resolved: `nyra`**
- ~~What specific reference direction should the Pinterest board have?~~ → **Resolved: Mixed European-American, All-American attractive archetype**
- Which LoRA training tool to commit to (Kohya, AI-Toolkit, Flux Gym, SimpleTuner)?
- Should the QC pipeline be extended with skin-tone consistency auto-check? → **Resolved: Built into `pinterest_qc_pro.py` via CIELAB + ΔE**
- Should we add a face-swap or face-replacement step for images that are good but have slightly wrong faces?

---

## 12. Key Constraints & Boundaries

- **Fully local** — no cloud APIs for generation or QC (except optional downloads for model weights)
- **Offline-friendly** — once models are downloaded, everything runs locally
- **One person** — the dataset is for a single character identity
- **AI-generated subject** — not a real person (no privacy concerns)
- **Max realism** — quality over speed, leveraging 96GB VRAM
- **Pilot-driven** — tune on small batch before scaling
- **Practical, not theoretical** — every step should be executable, not just a concept

---

*End of consolidated project context. Use this file to continue work in any future session.*
