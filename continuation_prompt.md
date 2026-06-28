# Girl Character LoRA — Continuation Prompt for Kimi

> **Paste this entire block into a new Kimi chat to continue the project exactly where you left off.**

---

## USER BACKGROUND & WORKING STYLE

- **Name:** Saransh Kumar, Bengaluru, India
- **Role:** AI engineer / ML practitioner / cybersecurity enthusiast
- **Technical stack:** Python, Bash, CUDA, PyTorch, GPU workflows, AI model hosting, quantization, automation, debugging
- **Tools used:** Kaggle notebooks, NVIDIA NIM, Lightning AI, Hugging Face, n8n, Apify, Playwright, Kali Linux, VS Code
- **Working style:** Highly iterative — prototype, test, inspect failures, tune, repeat
- **Preferences:** Actionable, concise, engineering-focused answers tuned for real implementation, not theory

## PROJECT CONTEXT

I am building a **local, fully automated QC pipeline for a female character LoRA dataset** using **ComfyUI + FLUX.2-dev full precision** on an **RTX PRO 6000 96GB VRAM**.

The character is **AI-generated** (not a real person). The goal is maximum realism with strict consistency across:
- face identity, hair, eyes, skin tone, face shape
- body shape, body proportions, realistic anatomy
- outfit variety, pose variety, lighting variety, background variety

The trained LoRA must work for **portraits, extreme close-ups, AND full-body shots** without losing identity.

## HARDWARE & ENVIRONMENT
- GPU: RTX PRO 6000 96GB VRAM
- Model: FLUX.2-dev full precision (`flux2-dev.safetensors`)
- Tool: ComfyUI (fully local)
- QC: Local Python pipeline (`lora_qc.py`) — offline-friendly

## DATASET PLAN
- Target: 120–200 images (max ~300 to avoid overfitting)
- Distribution: ~20% close-ups, ~30% portraits, ~30% half-body, ~15% full-body, ~5% contextual
- Resolution: 1024×1024 (square) or 1024×1280 (portrait)
- Outfit variety: casual, dresses, fitted/form-fitting, streetwear, minimal baseline — change every image
- Lighting & backgrounds: varied every time
- Trigger word: TBD (needs to be short, unique, distinct from common words)

## QC PIPELINE (Already Built)
- Script: `lora_qc.py` (local Python, works offline)
- Gates: Identity → Body/Proportion → Hand/Anatomy → Quality (Blur/Exposure) → Deduplication → Consistency Auto-Eval
- Output: `pass/`, `review/`, `reject/` + `report.csv`
- Thresholds: centralized, tunable, pilot-driven tuning order: Identity → Blur → Exposure → Duplicates → Pose → Hands
- Hard reject only obvious failures; borderline → REVIEW; portrait/close-up pose softened; small/distant hands softened

## CAPTIONING STRATEGY
- Describe: pose, outfit, environment, lighting, camera style, expression, framing
- AVOID: permanent identity traits (eye color, hair color, face shape) — trigger word carries these
- Trigger word should be first token in every caption

## TRAINING PLAN
- Tool: TBD (Kohya_ss, AI-Toolkit, Flux Gym, or SimpleTuner)
- Settings: Rank 16–32, Alpha 8–16, LR 1e-4–5e-4, Repeats 10–20, Batch 2–4, Steps 1000–3000, Resolution 1024×1024
- Regularization: 10–20 generic person images to prevent overfitting
- Random cropping during training for face + body balance

## CURRENT STATUS
- QC pipeline (`lora_qc.py` + `docs.md`) is complete and saved locally
- Script has been hardened with: duplicate-chain fix, false reject mitigation, portrait/close-up softening, small/distant hand handling, multi-face detection, low identity → review routing, robust directory scanning
- Gemini Deep Research prompt has been drafted for ComfyUI workflow research
- Character design direction is Pinterest-style reference moodboard (face, hair, outfit, pose, lighting, mood)
- Next step: **define the character, run pilot batch (20–30 images), QC, tune, scale**

## FILES IN WORKSPACE
- `project_master_context.md` — complete consolidated project document with user background
- `perplexity_raw_content.md` — raw text dump from both Perplexity chats
- `qc_docs.md` — QC documentation & tuning checklist
- `qc_lora_qc.py` — full Python QC pipeline script
- `continuation_prompt.md` — this file

## WHAT I NEED HELP WITH NOW
[PICK ONE OR MORE AND DELETE THE REST]

1. **Character definition** — Help me define the specific visual identity of the girl (face, hair, body type, ethnicity, age, aesthetic style). Build a reference direction I can use for generation.
2. **Generation prompt design** — Write a set of ComfyUI-ready prompts for the pilot batch (20–30 images) with varied outfits, poses, lighting, and environments.
3. **ComfyUI workflow** — Build or research the exact node structure for FLUX.2-dev in ComfyUI for max realism, including sampler settings, prompts, optional quality enhancement nodes, and a reroll/selection strategy.
4. **QC pipeline refinement** — Add skin-tone consistency auto-check, face-shape embedding comparison, or body-proportion variance scoring to the existing `lora_qc.py`.
5. **LoRA training setup** — Recommend the best training tool and exact settings for this dataset type, including regularization strategy and overfitting prevention.
6. **Pinterest reference strategy** — Help me organize a reference collection by category and define what to look for in each category.
7. **Pilot batch execution** — Walk me through running the first 20–30 image generation, QC processing, and threshold tuning cycle.
8. **Auto-evaluation pipeline** — Build an automatic consistency checker that scores face similarity, hair consistency, eye consistency, skin tone, and body shape across all dataset images and flags outliers for review.

## IMPORTANT RULES
- Everything must be fully local and offline-friendly after model download.
- The workflow must be practical and executable, not theoretical.
- Maximum realism is the top priority.
- The pilot batch (20–30 images) must be run and tuned before scaling to the full dataset.
- The character is AI-generated, not a real person.
- Do not suggest cloud-based generation or QC services.

---

*Paste this into a new Kimi chat and tell it which section(s) you want to work on next.*
