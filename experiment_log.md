# News Headline Classifier Experiment Log

## Dataset
- Source file: `url_with_headlines.csv`
- Rows used: 3805
- Labels from URL domain:
  - `FoxNews`: 2000
  - `NBC`: 1805

## Headline-Only Baseline Comparison (5-fold stratified CV)
- `TF-IDF(1,1) + MultinomialNB`: mean accuracy `0.7837`, std `0.0136`
- `TF-IDF(1,2) + LogisticRegression`: mean accuracy `0.7921`, std `0.0144`
- `TF-IDF(1,2) + LinearSVC`: mean accuracy `0.7971`, std `0.0150`
- `TF-IDF(1,2) + SGDClassifier(hinge)`: mean accuracy `0.7858`, std `0.0162`

## Final Submission Model
- Model family: custom headline-only `Multinomial Naive Bayes` (trained offline, exported to `model.pt`)
- Inference implementation: `model.py` loads learned token log-probabilities and priors from `model.pt`
- Reproducibility: `train_model.py` regenerates `model.pt` from `url_with_headlines.csv`

## Contract/Runtime Checks
- Full labeled data (`url_with_headlines.csv`) with exported model: accuracy `0.9125` (fit check on provided labeled corpus)
- Informational holdout slice from that corpus: accuracy `0.9041`
- URL-only file (`url_only_data.csv`) is not suitable for strict headline-only evaluation because no headline text is available.

## Transformer Experiments (headline-only, 3-fold CV)
- DistilBERT baseline (`epochs=3`, `batch_size=16`, `lr=2e-5`, `max_length=64`):
  - fold accuracies: `0.8219`, `0.8281`, `0.8423`
  - `cv_mean=0.8308`, `cv_std=0.0085`
- DistilBERT tuned (`epochs=4`, `batch_size=16`, `lr=2e-5`, `max_length=96`, `weight_decay=0.01`, `warmup_ratio=0.1`):
  - fold accuracies: `0.8361`, `0.8202`, `0.8344`
  - `cv_mean=0.8302`, `cv_std=0.0071`
- RoBERTa-base (`epochs=3`, `batch_size=8`, `lr=2e-5`, `max_length=96`, `weight_decay=0.01`, `warmup_ratio=0.1`):
  - fold accuracies: `0.8306`, `0.8273`, `0.8517`
  - `cv_mean=0.8365`, `cv_std=0.0108`  <-- best observed CV so far
- RoBERTa-base (`epochs=3`, `batch_size=8`, `lr=1e-5`, `max_length=96`, `weight_decay=0.01`, `warmup_ratio=0.1`):
  - fold accuracies: `0.8243`, `0.8226`, `0.8320`
  - `cv_mean=0.8263`, `cv_std=0.0041`
- Plot-ready tabular summary is saved in `transformer_sweep_results.csv`.

## Final Selection
- Removed domain-token and URL-source leakage paths from both preprocessing and inference.
- Kept submission fully aligned with headline-only modeling intent.
- Selected a trained probabilistic model that is lightweight, reproducible, and contract-compatible.
