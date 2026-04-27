import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def read_headline_dataset(csv_path: Path) -> List[Tuple[List[str], str]]:
    docs: List[Tuple[List[str], str]] = []
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

            docs.append((tokenize(headline), label))
    return docs


def train_nb(docs: List[Tuple[List[str], str]], min_df: int = 2) -> dict:
    vocab_counter: Counter = Counter()
    fox_counts: Counter = Counter()
    nbc_counts: Counter = Counter()
    fox_docs = 0

    for tokens, label in docs:
        vocab_counter.update(tokens)
        if label == "FoxNews":
            fox_docs += 1
            fox_counts.update(tokens)
        else:
            nbc_counts.update(tokens)

    vocab = sorted([word for word, df in vocab_counter.items() if df >= min_df])
    vocab_index = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    fox_total = sum(fox_counts[word] for word in vocab)
    nbc_total = sum(nbc_counts[word] for word in vocab)
    alpha = 1.0

    log_prob_fox = [0.0] * vocab_size
    log_prob_nbc = [0.0] * vocab_size

    for word, i in vocab_index.items():
        log_prob_fox[i] = math.log((fox_counts[word] + alpha) / (fox_total + alpha * vocab_size))
        log_prob_nbc[i] = math.log((nbc_counts[word] + alpha) / (nbc_total + alpha * vocab_size))

    log_prior_fox = fox_docs / len(docs)
    unk_log_prob_fox = math.log(alpha / (fox_total + alpha * vocab_size))
    unk_log_prob_nbc = math.log(alpha / (nbc_total + alpha * vocab_size))

    return {
        "model_type": "headline_multinomial_nb",
        "vocab": vocab,
        "log_prob_fox": torch.tensor(log_prob_fox, dtype=torch.float32),
        "log_prob_nbc": torch.tensor(log_prob_nbc, dtype=torch.float32),
        "log_prior_fox": torch.tensor(log_prior_fox, dtype=torch.float32),
        "unk_log_prob_fox": torch.tensor(unk_log_prob_fox, dtype=torch.float32),
        "unk_log_prob_nbc": torch.tensor(unk_log_prob_nbc, dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train headline-only Naive Bayes model and export model.pt")
    parser.add_argument("--train_csv", default="url_with_headlines.csv")
    parser.add_argument("--output", default="model.pt")
    parser.add_argument("--min_df", type=int, default=2)
    args = parser.parse_args()

    docs = read_headline_dataset(Path(args.train_csv))
    if not docs:
        raise ValueError("No valid FoxNews/NBC rows found in training CSV.")

    state = train_nb(docs, min_df=args.min_df)
    torch.save(state, args.output)

    print(f"Saved {args.output}")
    print(f"Rows: {len(docs)}")
    print(f"Vocab size: {len(state['vocab'])}")


if __name__ == "__main__":
    main()
