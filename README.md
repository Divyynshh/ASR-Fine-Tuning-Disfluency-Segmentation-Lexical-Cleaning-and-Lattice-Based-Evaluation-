# 🎙️ Hindi ASR: Fine-Tuning, Disfluency Detection, Lexical Cleaning & Lattice WER

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper--small-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**A complete Hindi Speech AI pipeline: ASR fine-tuning · disfluency segmentation · spelling classification · fair evaluation**

*46% relative WER reduction on conversational Hindi — from 62.3% → 33.6%*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Project Structure](#-project-structure)
- [Question 1 — Whisper Fine-Tuning](#-question-1--whisper-small-fine-tuning)
- [Question 2 — Disfluency Detection](#-question-2--disfluency-detection--audio-segmentation)
- [Question 3 — Spelling Classification](#-question-3--hindi-spelling-classification)
- [Question 4 — Lattice-Based WER](#-question-4--lattice-based-wer-with-reference-correction)
- [Setup & Usage](#-setup--usage)
- [Author](#-author)

---

## 🔍 Overview

This project implements a four-part Hindi Speech AI pipeline on ~10 hours of real conversational Hindi audio (104 recordings, 7–20 minutes each). The dataset comes from JoshTalks and is accessed via Google Cloud Storage.

| Task | What it does |
|------|-------------|
| **Q1** | Fine-tune Whisper-small on Hindi; evaluate WER vs baseline |
| **Q2** | Detect and clip speech disfluencies (fillers, repetitions, false starts) |
| **Q3** | Classify ~1,77,000 unique transcription words as correctly/incorrectly spelled |
| **Q4** | Build a word lattice across 5 ASR models; correct unfair WER from reference errors |

---

## 🏆 Results at a Glance

### Q1 — Word Error Rate

| Model | WER (%) | Relative Change |
|-------|---------|-----------------|
| Whisper-small (Pretrained Baseline) | 62.30 | — |
| **Whisper-small (Fine-tuned, ~10h Hindi)** | **33.57** | **↓ 46.1%** |

**Training curve:**

| Step | Train Loss | Val Loss | Val WER (%) |
|------|-----------|---------|------------|
| 187 | 0.432 | 0.398 | 42.94 |
| 374 | 0.242 | 0.316 | 37.81 |
| 561 | 0.252 | 0.294 | 33.70 |
| 748 | 0.176 | 0.292 | 34.19 |
| **935** | **0.192** | **0.289** | **33.57** ✅ |

> Best model selected at minimum validation WER. Training: ~90 min on T4 GPU.

### Q2 — Disfluency Detection

7 categories detected across 7 regex pattern groups:
`filler_uh` · `filler_umm` · `filler_hindi` · `repetition` · `false_start` · `prolongation` · `hesitation`

Output: CSV with per-occurrence rows + clipped `.wav` files per segment.

### Q3 — Spelling Classification

| Category | Count | % |
|----------|-------|---|
| Correctly spelled | ~1,33,000 | ~75% |
| Incorrectly spelled | ~44,000 | ~25% |
| **Total unique words** | **~1,77,000** | 100% |

### Q4 — Lattice WER (Demo)

| Model | Standard WER (%) | Lattice-Corrected WER (%) | Δ |
|-------|-----------------|--------------------------|---|
| Model_A | 20.0 | 0.0 | **−20** |
| Model_B | 20.0 | 0.0 | **−20** |
| Model_C | 20.0 | 20.0 | 0 |
| Model_D | 20.0 | 0.0 | **−20** |
| Model_E | 40.0 | 40.0 | 0 |

> Models A/B/D were correct but penalized by a reference error. The lattice detects 80% consensus and corrects the effective reference.

---

## 📁 Project Structure

```
hindi-asr-pipeline/
├── Hindi_ASR_Complete_Colab_Final.ipynb   # Full notebook (all 4 questions)
├── outputs/
│   ├── q1_wer_results.csv                 # WER table: baseline vs fine-tuned
│   ├── q1_training_curve.csv              # Step-by-step training metrics
│   ├── q2_disfluency_dataset.csv          # Disfluency occurrences with timestamps
│   ├── q3_spelling_classification.csv     # ~1,77,000 words labeled correct/incorrect
│   ├── q4_lattice_wer_results.csv         # Standard vs lattice-corrected WER per model
│   └── q4_lattice_visualization.png       # WER comparison + lattice agreement chart
├── disfluency_clips/                      # Segmented .wav clips (one per disfluency row)
└── README.md
```

---

## 🤖 Question 1 — Whisper-Small Fine-Tuning

### Architecture

```
Full Recording (7–20 min WAV)
        │
        ▼
Transcription JSON (segments with timestamps)
        │
        ├── segment 0: start=0.11s  end=14.42s  text="मैं यहाँ..."
        ├── segment 1: start=14.5s  end=27.3s   text="तो जो बात..."
        └── segment N: ...
        │
        ▼ librosa.load(offset=start_sec, duration=seg_dur)
Audio Slice [16kHz, mono, normalized]
        │
        ▼ WhisperFeatureExtractor
80-channel Log-Mel Spectrogram (shape: 80 × 3000)
        │
        ▼ Whisper Encoder (Transformer)
Encoder Hidden States
        │
        ▼ Whisper Decoder (Autoregressive)
Token IDs → Devanagari Text
```

### Data Preprocessing Pipeline

**URL pattern discovery** — The CSV contained broken GCS paths. The working pattern was:
```
https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_transcription.json
https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav
```
`folder_id` is extracted from the original URL via regex: `/hi/(\d+)/`

**Text normalization:**
1. Unicode NFC — collapses multi-byte Devanagari to canonical form
2. Control character removal — strips ASCII 0x00–0x1F embedded in transcription files
3. Whitespace collapsing — normalizes double spaces and trims edges

**Audio normalization:**
- Resample to 16,000 Hz (Whisper requirement)
- Convert stereo → mono
- Amplitude normalize to [-1, 1]

**Segment filtering:** 0.5s < duration < 28s → yields ~5,571 valid segments from 104 recordings

### Training Configuration

| Hyperparameter | Value | Why |
|---------------|-------|-----|
| Model | whisper-small | 244M params, multilingual |
| Batch size | 16 | Fits T4 with FP16 |
| Max steps | `(n_train // 16) × 3` | Scales to actual dataset |
| Learning rate | 1e-5 | Conservative for fine-tuning |
| FP16 | ✅ | 2× throughput on T4 |
| Gradient checkpointing | ✅ | Reduces VRAM |
| Best model | Min val WER | `load_best_model_at_end=True` |

### Key Implementation Notes

- **Segment-based loading**: `librosa.load(offset=start_sec, duration=...)` reads only the needed window from the full WAV — no pre-cutting thousands of audio files to disk
- **Feature caching**: Extracted features saved to disk with `Dataset.save_to_disk()` — subsequent runs load instantly; survives Colab session resets
- **API compatibility**: Uses `eval_strategy` (not `evaluation_strategy`) and `processing_class` (not `tokenizer`) for transformers ≥ 4.41

---

## 🗣️ Question 2 — Disfluency Detection & Audio Segmentation

### Detection Approach

Text-based regex pattern matching on transcription text, applied at segment granularity. Since human transcriptions already encode what was said, acoustic detection is not required — the timestamps from the JSON tell us exactly where in the audio each disfluency occurs.

### Pattern Categories

| Category | Regex Pattern (sample) | Example Matches |
|----------|----------------------|-----------------|
| `filler_uh` | `\buh+\b`, `\bah+\b` | uh, uhh, ah |
| `filler_umm` | `\bumm+\b`, `\bum+\b` | um, umm, hmm |
| `filler_hindi` | `\bमतलब\b`, `\bयानी\b` | मतलब, यानी, हाँ तो |
| `repetition` | `\b(\w+)\s+\1\b` | main main, jo jo |
| `false_start` | `\w+[-–—]\s` | actually— , jo— |
| `prolongation` | `([char])\1{2,}` | sooooo, nahiiiii |
| `hesitation` | `\ber+\b`, `\bachha+\b` | er, erm, achha |

All 20 patterns compiled once with `re.IGNORECASE | re.UNICODE`.

### Audio Clipping

```python
audio = AudioSegment.from_file(recording.wav)
clip  = audio[start_ms - 100 : end_ms + 100]   # 100ms padding each side
clip  = clip.set_channels(1).set_frame_rate(16000).set_sample_width(2)
clip.export(clip_name, format="wav")
```

### Output Schema

| Column | Description |
|--------|-------------|
| `recording_id` | Source recording ID |
| `segment_id` | Segment index within recording |
| `disfluency_type` | Category name |
| `matched_text` | Exact regex match |
| `segment_text` | Full segment transcription |
| `segment_start_sec` | Start timestamp (seconds) |
| `segment_end_sec` | End timestamp (seconds) |
| `clip_filename` | Saved `.wav` filename |
| `clip_audio_link` | Relative path to clip |

---

## 🔤 Question 3 — Hindi Spelling Classification

### The Challenge

~1,77,000 unique words from conversational Hindi transcriptions. Some are typos; some are rare but valid words; some are English words written in Devanagari (per transcription guidelines — these are **correct**).

### Five-Layer Classification

```
Word
 │
 ├─ Layer 1: Unicode validation ──── invalid chars? → INCORRECT
 │
 ├─ Layer 2: Devanagari morphology ── double matra / dangling halant? → INCORRECT
 │
 ├─ Layer 3: Dictionary lookup ────── in IndicNLP + iNLTK vocab? → CORRECT
 │
 ├─ Layer 4: Loanword list ──────────── known Devanagari loanword? → CORRECT
 │           (कंप्यूटर, मोबाइल, इंटरनेट ...)
 │
 ├─ Layer 5: SymSpell edit-distance ── 1 edit from dict word? → INCORRECT
 │           (मैने → मैंने, करत → करता)
 │
 └─ Default ─────────────────────────── OOV but structurally valid → CORRECT
```

**Why conservative default?** Hindi has a vast vocabulary including regional, colloquial, and domain-specific words not in standard dictionaries. OOV ≠ wrong.

**Why SymSpell over brute-force Levenshtein?** At 100K+ dictionary scale, brute-force lookup is O(n) per query. SymSpell's symmetric delete algorithm achieves O(1) average by precomputing deletion variants — lookup across 177K words completes in minutes rather than hours.

---

## 📊 Question 4 — Lattice-Based WER with Reference Correction

### The Problem with Standard WER

When a human reference contains an error, every model that transcribes correctly gets penalized. Standard WER cannot distinguish "model was wrong" from "reference was wrong."

### Word Lattice Construction

```
                    ┌─ "aaj" (w=0.80) ──────────────┐
Human ref: "main kal school gaya tha"                │
                    │                                 ▼
Position 2: ────────┤         Effective ref: "main aaj school gaya tha"
                    │         (ref corrected: 4/5 models agree on "aaj")
                    └─ "kal" (w=0.20) ──────────────┘
```

**Algorithm:**
1. Align each of the 5 model hypotheses to the reference via Levenshtein DP
2. At each alignment position, count the distribution of model predictions
3. If ≥ 60% of models agree on word X ≠ reference → `effective_ref[pos] = X`
4. Score each model against `effective_ref` (not raw reference)

### Why Word-Level Alignment?

| Unit | Problem |
|------|---------|
| Subword | Different models use different tokenizers → incomparable |
| Phrase | Over-aggregates; can't distinguish S/D/I within phrase |
| **Word** ✅ | Universal unit; directly maps to WER; interpretable |

### Threshold Justification

60% (3 of 5 models) was chosen as the consensus threshold. Lower thresholds risk overriding correct references when two models happen to share the same error. Higher thresholds (e.g., 80%) would miss many genuine reference errors in practice.

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
# GPU runtime recommended for Q1 (T4 or better)
# CPU sufficient for Q2, Q3, Q4
```

### Run in Google Colab

1. Open `Hindi_ASR_Complete_Colab_Final.ipynb` in Colab
2. Set runtime to **GPU (T4)** — `Runtime → Change runtime type`
3. Run **Shared Setup** cells first (installs all packages)
4. Each question section is independently runnable after setup

### Dependencies

```bash
pip install transformers datasets evaluate jiwer accelerate
pip install librosa soundfile pydub symspellpy
pip install pandas requests tqdm matplotlib
apt-get install ffmpeg
```

### Data Access

The dataset requires the `FT_Data.csv` metadata file. Audio and transcription files are downloaded automatically from GCS during execution using the URL pattern described in Q1.

---

## 👤 Author

**Divyansh Bansal**
Prompt Engineer · LLM Engineer · AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Divyansh_Bansal-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/divyansh-bansal)
[![GitHub](https://img.shields.io/badge/GitHub-Divyynshh-181717?style=flat-square&logo=github)](https://github.com/Divyynshh)
[![Email](https://img.shields.io/badge/Email-bansaldivyansh001@gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:bansaldivyansh001@gmail.com)

> B.Tech CSE · ITM University Gwalior · Expected June 2026
> Previously: Prompt Engineer @ Frostrek AI · AI Annotator @ Innodata

---

<div align="center">
<sub>Built with 🎙️ Whisper · 🤗 HuggingFace · 🐍 Python · ☁️ Google Colab</sub>
</div>
