# CIS 4190 — Project B: News headline source classifier

Binary classification: **Fox News** vs **NBC** from headline text (labels derived from article URL domain in the training CSV).

## What to submit (leaderboard)

Upload exactly:

- `preprocess.py` — must define `prepare_data(csv_path) -> (X, y)`
- `model.py` — `Model` / `get_model()` and `predict(batch)`
- `model.pt` — weights (large; keep locally or in team storage; not required in git)

## Repository layout

| Path | Role |
|------|------|
| `preprocess.py` | CSV → headline strings + string labels |
| `model.py` | Loads checkpoint; RoBERTa-style classifier + optional sklearn fallback |
| `train_distilbert.py` | Fine-tune Hugging Face models (CUDA → MPS → CPU) |
| `export_submission_roberta.py` | Pack HF artifacts + tokenizer bytes into `model.pt` |
| `train_model.py` | Classical baseline (TF-IDF + LinearSVC); small `model.pt` for debugging |
| `eval_project_b.py` | Local sanity check (mirrors backend flow) |
| `eval_distilbert_cv.py` | Stratified K-fold CV for transformers |
| `run_transformer_sweep.py` | Batch CV configs → CSV |
| `experiment_log.md` | Early baselines and dataset notes |
| `model_iteration_log.md` | Full iteration story, packaging, leaderboard notes |
| `transformer_sweep_results.csv` | Tabular CV sweep output |
| `url_with_headlines.csv` | Labeled training data (`url`, `headline`) |
| `url_only_data.csv` | URL-only examples (optional / scraping workflows) |
| `requirements.txt` | Pip deps for a fresh venv |

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `model.pt` next to `model.py`, then:

```bash
python eval_project_b.py --model model.py --preprocess preprocess.py --csv url_with_headlines.csv --weights model.pt
```

## Re-train and export transformer weights

```bash
python train_distilbert.py --train_csv url_with_headlines.csv --output_dir submission_roberta_artifacts \
  --model_name roberta-base --epochs 3 --batch_size 8 --lr 2e-5 --max_length 128 \
  --weight_decay 0.01 --warmup_ratio 0.1 --seed 42
python export_submission_roberta.py --artifacts_dir submission_roberta_artifacts --output model.pt
```

On GPU instances, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org) if `torch.cuda.is_available()` is false after `pip install torch`.

## Report

Use **`REPORT_MATERIALS.md`** as the single consolidated source for the write-up: **§2 iteration narrative**, tables, caveats, and outline. Raw logs: `experiment_log.md`, `model_iteration_log.md` (older notes; some lines are superseded — the report doc reconciles them).
