# Report materials — Project B (News headline source classifier)

**Use this file as your primary source for the write-up.** It consolidates `experiment_log.md`, `model_iteration_log.md`, and the current codebase. You do **not** need to paste those raw logs into the report; cite numbers and methods from here, and attach or link the repo as needed.

---

## 1. Problem statement (what to say in the report)

- **Task:** Classify which **news outlet** a headline came from: **Fox News** vs **NBC**, using **headline text**.
- **Training data:** CSV with `url` and `headline`. **Class labels are derived from the URL domain** (`foxnews.com` → FoxNews, `nbcnews.com` → NBC). The model must **not** use URL text as an input feature — only the headline string is passed to the classifier (headline-only modeling).
- **Evaluation:** Course leaderboard on a hidden validation split (`url_val` in the UI). Local checks use `eval_project_b.py` to mimic the backend contract (`prepare_data` → `predict`).

---

## 2. Dataset summary (for Methods / Data)

| Item | Value |
|------|--------|
| Primary file | `url_with_headlines.csv` |
| Rows used (labeled Fox/NBC) | 3805 |
| FoxNews | 2000 |
| NBC | 1805 |
| Optional | `url_only_data.csv` (URLs only — not usable for headline-only model training without headlines) |

**Preprocessing (current):** `prepare_data(csv_path)` reads the CSV, cleans headline text (HTML unescape, whitespace, quote stripping), lowercases/normalizes for classical baselines in older experiments; **the submitted transformer pipeline uses raw headline casing** after cleaning where applicable — a key finding was that **aggressive lowercasing hurt RoBERTa**.

---

## 3. Course / submission constraints (short paragraph)

- `preprocess.py`: `prepare_data(csv_path) -> (X, y)` with string labels `FoxNews` / `NBC` (or compatible).
- `model.py`: `get_model()` or default `Model()`; backend calls `predict(batch)`.
- `model.pt`: optional; loaded as state dict into the model. Your final submission packs **RoBERTa weights + tokenizer bytes** into one file so only **three files** need uploading: `preprocess.py`, `model.py`, `model.pt`.

---

## 4. Baselines and classical models (Results / Related work)

### 4.1 Early TF–IDF baselines (5-fold stratified CV, headline-only)

From `experiment_log.md`:

| Model | CV mean accuracy | CV std |
|-------|------------------|--------|
| TF-IDF (1,1) + Multinomial NB | 0.7837 | 0.0136 |
| TF-IDF (1,2) + Logistic Regression | 0.7921 | 0.0144 |
| TF-IDF (1,2) + Linear SVC | **0.7971** | 0.0150 |
| TF-IDF (1,2) + SGD (hinge) | 0.7858 | 0.0162 |

### 4.2 Later classical sweep (char n-grams + LinearSVC, 5-fold CV)

From iteration log:

| Config | CV mean | CV std |
|--------|---------|--------|
| char_wb (3–5), LinearSVC | 0.8152 | 0.0112 |
| char_wb (2–6), LinearSVC | **0.8208** | 0.0128 |
| hybrid word (1–2) + char (3–5) | 0.8179 | 0.0113 |

**Takeaway for report:** Classical models cap out around **~0.82** CV on this data; moving to transformers was necessary for a large jump.

---

## 5. Transformer experiments (main Results section)

### 5.1 Early transformer CV (3-fold; some runs used `max_length` 64–96)

From `experiment_log.md`:

| Setting | CV mean | Notes |
|---------|---------|--------|
| DistilBERT, baseline hparams | 0.8308 | |
| DistilBERT, tuned | 0.8302 | |
| RoBERTa-base, `max_length=96` | **0.8365** | best in that sweep |

### 5.2 Refined RoBERTa study (headline-only, preserve case)

Key result: **do not lowercase** headlines for RoBERTa.

| Configuration | CV mean | CV std |
|----------------|---------|--------|
| RoBERTa-base, 3 epochs, bs 8, lr 2e-5, max_len **128**, case preserved | **0.8526** | 0.0169 |
| RoBERTa-base, max_len 192 | 0.8470 | 0.0106 |
| RoBERTa-base, 4 epochs, lr 1.5e-5 | 0.8463 | 0.0068 |
| RoBERTa-large (same budget) | ~0.799 | unstable |
| DeBERTa-v3-base (tuned attempt) | ~0.822 | |
| BERT-base-cased | ~0.835 | |

**Report sentence:** Larger models did not automatically win; **RoBERTa-base with conservative fine-tuning and preserved casing** gave the best cross-validated headline-only performance in this project.

### 5.3 Final training command (reproducibility)

Fine-tune on full training CSV, then export:

```bash
python train_distilbert.py \
  --train_csv url_with_headlines.csv \
  --output_dir submission_roberta_artifacts \
  --model_name roberta-base \
  --epochs 3 --batch_size 8 --lr 2e-5 --max_length 128 \
  --weight_decay 0.01 --warmup_ratio 0.1 --seed 42

python export_submission_roberta.py \
  --artifacts_dir submission_roberta_artifacts \
  --output model.pt
```

(Device order in code: **CUDA → MPS → CPU** for cloud vs Mac.)

---

## 6. Leaderboard vs local metrics (important caveat)

| Metric | Typical value | Interpretation |
|--------|----------------|----------------|
| Leaderboard `url_val` (earlier classical / brittle prep) | ~**0.799** | Hidden distribution; preprocessing bugs (e.g. live crawling) hurt badly |
| Leaderboard after RoBERTa + stable `prepare_data` | ~**0.837** (your “big-test” style run) | Meaningful gain from transformers + deterministic pipeline |
| Local `eval_project_b.py` on **same** CSV used for training | ~**0.97** | **Not** an unbiased generalization estimate — near in-distribution fit on labeled corpus |

**For the report:** Always distinguish **hidden-test leaderboard accuracy** from **in-sample local accuracy**. Discuss **domain shift** and why CV (~0.85) tracks more realistically than ~0.97 local checks.

---

## 7. Engineering lessons (Discussion)

1. **Deterministic preprocessing:** `prepare_data` must not depend on live scraping for leaderboard reliability.
2. **Casing:** Lowercasing removed useful signal for pretrained tokenizers; **preserve case** for RoBERTa.
3. **Submission packaging:** Single `model.pt` bundling weights + tokenizer files satisfies **three-file-only** upload limits (~480MB — acceptable if the platform allows).
4. **Latency:** Transformer inference is slower (~**8 ms**/example on leaderboard vs ~**0.5 ms** for classical) — trade accuracy for speed.
5. **Git:** Large weights and `.venv` should stay out of version control; use `.gitignore` and team file sharing for `model.pt`.

---

## 8. Suggested report outline (fill with prose)

1. **Introduction:** Task, motivation, headline-only constraint.
2. **Data:** Source, label derivation, class balance, cleaning.
3. **Methods:** Baselines (TF-IDF, char n-grams), then RoBERTa fine-tuning; loss, optimizer (AdamW), schedule, hyperparameters table.
4. **Experiments:** CV protocol (stratified K-fold), ablations (casing, length, model size).
5. **Results:** Table of CV + **leaderboard**; confusion / error analysis if you have time.
6. **Discussion:** Limitations (topic drift, ambiguous headlines), ethics (source ≠ ideology), future work (ensembling).
7. **Reproducibility:** Point to `README.md`, `train_distilbert.py`, `export_submission_roberta.py`, `eval_project_b.py`.

---

## 9. Files in repo (what each is for)

| File | Purpose |
|------|---------|
| `REPORT_MATERIALS.md` | **This document** — report source of truth |
| `experiment_log.md` | Original baseline + early transformer notes (some “final model” lines are **outdated**) |
| `model_iteration_log.md` | Chronological engineering log (includes superseded ideas — use **Section 1–8 here** for consistency) |
| `README.md` | Quick start and commands |
| `transformer_sweep_results.csv` | Numeric sweep snapshot |

---

## 10. One-sentence “final model” blurb (abstract-ready)

> We fine-tuned **RoBERTa-base** as a binary sequence classifier on **headline-only** text (labels from publisher URL), using **case-preserving** preprocessing, **AdamW** with linear warmup, and exported weights plus tokenizer into a single **`model.pt`** for leaderboard submission, improving hidden validation accuracy from roughly **0.80** to **~0.84** while accepting higher per-example latency.

*(Adjust the exact leaderboard number to match your final submitted alias screenshot.)*
