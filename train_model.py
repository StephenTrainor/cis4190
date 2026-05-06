import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return normalized if normalized else "[EMPTY_HEADLINE]"


def read_headline_dataset(csv_path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = str(row.get("url") or row.get("\ufeffurl") or "").lower()
            headline = str(
                row.get("headline")
                or row.get("scraped_headline")
                or row.get("alternative_headline")
                or row.get("title")
                or ""
            )

            if "foxnews.com" in url:
                label = "FoxNews"
            elif "nbcnews.com" in url:
                label = "NBC"
            else:
                continue

            headline_norm = normalize_text(headline)
            texts.append(headline_norm)
            labels.append(label)
    return texts, labels


def to_2d_text_column(values):
    return [[v] for v in values]


def build_candidates() -> List[Tuple[str, Pipeline]]:
    candidates: List[Tuple[str, Pipeline]] = []

    candidates.append(
        (
            "char_3_5_svc",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=2,
                            sublinear_tf=True,
                            max_features=180000,
                        ),
                    ),
                    ("clf", LinearSVC(C=1.0, random_state=42)),
                ]
            ),
        )
    )

    candidates.append(
        (
            "char_2_6_svc",
            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(2, 6),
                            min_df=2,
                            sublinear_tf=True,
                            max_features=220000,
                        ),
                    ),
                    ("clf", LinearSVC(C=0.7, random_state=42)),
                ]
            ),
        )
    )

    hybrid_features = ColumnTransformer(
        transformers=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                ),
                0,
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=180000,
                    sublinear_tf=True,
                ),
                0,
            ),
        ],
        remainder="drop",
    )
    candidates.append(
        (
            "hybrid_word12_char35_svc",
            Pipeline(
                [
                    ("reshape", FunctionTransformer(to_2d_text_column, validate=False)),
                    ("features", hybrid_features),
                    ("clf", LinearSVC(C=0.8, random_state=42)),
                ]
            ),
        )
    )

    return candidates


def train_best_classical_model(texts: List[str], labels: List[str]) -> Tuple[str, Pipeline, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name = ""
    best_model = None
    best_score = -1.0

    for name, pipeline in build_candidates():
        scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="accuracy", n_jobs=1)
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        print(f"candidate={name} cv_mean={mean_score:.6f} cv_std={std_score:.6f}")
        if mean_score > best_score:
            best_name = name
            best_model = pipeline
            best_score = mean_score

    assert best_model is not None
    best_model.fit(texts, labels)
    return best_name, best_model, best_score


def package_pipeline(pipeline: Pipeline, model_name: str) -> dict:
    blob = pickle.dumps(pipeline)
    return {
        "model_type": model_name,
        "checkpoint_probe": torch.ones(1, dtype=torch.float32),
        "sklearn_pipeline": torch.tensor(list(blob), dtype=torch.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train headline-only TF-IDF+LinearSVC model and export model.pt")
    parser.add_argument("--train_csv", default="url_with_headlines.csv")
    parser.add_argument("--output", default="model.pt")
    args = parser.parse_args()

    texts, labels = read_headline_dataset(Path(args.train_csv))
    if not texts:
        raise ValueError("No valid FoxNews/NBC rows found in training CSV.")

    best_name, pipeline, best_cv = train_best_classical_model(texts, labels)
    state = package_pipeline(pipeline, best_name)
    torch.save(state, args.output)

    print(f"Saved {args.output}")
    print(f"Rows: {len(texts)}")
    print(f"Selected model: {best_name}")
    print(f"Selected CV mean accuracy: {best_cv:.6f}")


if __name__ == "__main__":
    main()
