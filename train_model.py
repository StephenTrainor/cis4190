import argparse
import csv
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


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

            texts.append(normalize_text(headline))
            labels.append(label)
    return texts, labels


def train_best_classical_model(texts: List[str], labels: List[str]) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=150000,
                ),
            ),
            ("clf", LinearSVC(C=1.0, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def package_pipeline(pipeline: Pipeline) -> dict:
    blob = pickle.dumps(pipeline)
    return {
        "model_type": "headline_tfidf_char35_linearsvc",
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

    pipeline = train_best_classical_model(texts, labels)
    state = package_pipeline(pipeline)
    torch.save(state, args.output)

    print(f"Saved {args.output}")
    print(f"Rows: {len(texts)}")
    print("Model: TF-IDF(char_wb 3-5) + LinearSVC")


if __name__ == "__main__":
    main()
