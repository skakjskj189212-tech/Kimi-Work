# Character Definition: Nyra

> **Trigger word:** `nyra`  
> **Ethnicity:** Mixed European-American (Northern + Mediterranean blend) — the archetypal attractive, relatable "All-American" look trusted in US media and advertising.  
> **Purpose:** Primary identity carrier for the LoRA. Do NOT describe permanent traits in captions — the trigger word and anchor images carry these.

---

## Core Identity

| Attribute | Specification |
|-----------|---------------|
| **Age** | 24 years old |
| **Ethnicity** | Mixed European-American (Northern European + Mediterranean blend) |
| **Skin tone** | Warm beige with light olive undertone — consistent across all lighting conditions |
| **Face shape** | Heart-shaped to oval; high cheekbones; defined jawline; straight nose with slight upturn |
| **Eyes** | Almond-shaped; hazel-green; naturally expressive and bright |
| **Eyebrows** | Full; naturally arched; well-defined; light brown |
| **Lips** | Full; naturally pink-toned; balanced upper and lower |
| **Hair** | Long light brown to dark blonde; soft natural waves; voluminous; often falls over one shoulder |
| **Body type** | Athletic-slim; 5'6" equivalent proportions; natural waist-to-hip ratio; toned but not muscular |
| **Aesthetic** | Modern, approachable, confident — "girl next door" meets editorial; versatile across casual and formal |
| **Attractiveness anchor** | Relatable beauty standard seen in mainstream US media — fresh, healthy, photogenic, trustworthy |

---

## Reference Direction by Category

### Face Direction
- Heart-shaped to oval face with high cheekbones and defined jawline
- Straight nose with slight upturn at the tip
- Almond-shaped hazel-green eyes, bright and naturally expressive
- Full, naturally pink-toned lips
- Light brown naturally arched eyebrows
- Neutral expression should look confident and approachable, not blank
- Expression range: neutral confident, soft smile, intense gaze, playful, serious, laughing

### Hair Direction
- Long (mid-back length), light brown to dark blonde
- Soft natural waves, voluminous
- Often falls over left shoulder or across face in close-ups
- Can be styled: loose waves, half-up, ponytail, or slightly tousled
- Never: straightened flat, extreme colors, buzz cuts, or very short

### Outfit Direction (24 Variations for Full Dataset)
1. Fitted white ribbed t-shirt + light blue jeans
2. Cropped denim jacket + white tank top + dark jeans
3. Black leather jacket + dark fitted jeans
4. Cream knit sweater + minimal bottom
5. Emerald green turtleneck + dark trousers
6. Burgundy bodycon mini dress
7. Floral flowing midi dress
8. Black slip dress
9. Sheer black corset top + low-rise vintage jeans
10. Red silk camisole + dark trousers
11. Backless navy halter top + fitted skirt
12. White linen blouse + tailored trousers
13. Structured charcoal blazer + silk camisole
14. Cropped athletic top + joggers
15. Chunky knit cardigan + fitted turtleneck
16. Oversized hoodie + leggings
17. Simple black bikini (anatomical baseline)
18. Black sports bra + leggings (anatomical baseline)
19. Fitted white tank top + dark leggings (anatomical baseline)
20. Camel coat + black dress
21. Fitted black cocktail dress
22. Cropped graphic tee + high-waisted shorts
23. Light summer sundress
24. Nude sports bra + minimal bottom (anatomical baseline)

### Pose Direction
- Neutral frontal (40%)
- Three-quarter angle looking over shoulder (15%)
- Profile/side view (10%)
- Walking/mid-stride (10%)
- Sitting/leaning (10%)
- Hands on hips / arms crossed (10%)
- Extreme close-up face-only (5%)

### Lighting Direction
- Soft diffused natural light (30%)
- Golden hour warm sunlight (20%)
- Studio softbox/even lighting (15%)
- Moody dramatic side lighting (15%)
- Neon/ambient artificial (10%)
- Cold blue winter light (10%)

### Environment Direction
- Urban exteriors (streets, rooftops, parks)
- Indoor interiors (bedroom, kitchen, studio, bar)
- Natural settings (beach, park, courtyard)
- Professional settings (studio, office, gym)
- Night/ambient settings (jazz club, parking garage, rooftop)

### Mood / Style Direction
- Modern, confident, photorealistic
- Candid lifestyle aesthetic
- Editorial fashion
- Intimate/portrait style
- Street style

---

## Captioning Rules (Permanent vs. Variable)

### Permanent Identity Traits (NEVER put in captions)
- Eye color (hazel-green)
- Hair color (light brown to dark blonde)
- Face shape (heart-shaped/oval, high cheekbones, defined jawline)
- Skin tone (warm beige with light olive undertone)
- Body type (athletic-slim, waist-to-hip ratio)
- Nose shape (straight with slight upturn)
- Lip shape (full, naturally pink-toned)
- Eyebrow shape (full, naturally arched, light brown)

### Variable Traits (ALWAYS put in captions)
- Pose (standing, sitting, walking, looking over shoulder)
- Outfit (specific garment names)
- Environment (city street, bedroom, studio, beach)
- Lighting (golden hour, softbox, neon, moody)
- Camera style (85mm, 35mm, DSLR, film grain)
- Expression (smiling, neutral, confident, playful)
- Framing (close-up, portrait, half-body, full-body)

---

## Trigger Word Usage

| Phase | Usage |
|-------|-------|
| **Dataset Generation (ComfyUI)** | Do NOT use `nyra` in prompts. Use natural description + ReferenceLatent anchors. |
| **Captions for Training** | First token: `nyra`. Example: `nyra, standing on a city street, wearing a black leather jacket...` |
| **Inference (Post-Training)** | `nyra` carries all identity. Example: `Cinematic 35mm. nyra woman, standing...` |

---

*File: character_definition.md | Character: Nyra | Trigger: nyra*
