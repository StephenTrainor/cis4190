import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Tuple

import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def read_rows(csv_path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
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
                labels.append(0)
            elif "nbcnews.com" in url:
                labels.append(1)
            else:
                continue
            texts.append(normalize_text(headline))
    return texts, labels


class HeadlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def run_fold(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    model_name: str,
    weight_decay: float,
    warmup_ratio: float,
) -> float:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    train_ds = HeadlineDataset(train_enc, torch.tensor(train_labels, dtype=torch.long))
    val_ds = HeadlineDataset(val_enc, torch.tensor(val_labels, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu()
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.shape[0])
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-fold CV for headline-only DistilBERT.")
    parser.add_argument("--csv_path", default="url_with_headlines.csv")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    texts, labels = read_rows(Path(args.csv_path))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    scores: List[float] = []
    for fold_i, (tr, va) in enumerate(skf.split(texts, labels), start=1):
        train_texts = [texts[i] for i in tr]
        train_labels = [labels[i] for i in tr]
        val_texts = [texts[i] for i in va]
        val_labels = [labels[i] for i in va]
        acc = run_fold(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            model_name=args.model_name,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
        )
        scores.append(acc)
        print(f"fold={fold_i} acc={acc:.6f}")

    print(f"cv_mean={mean(scores):.6f}")
    print(f"cv_std={pstdev(scores):.6f}")


if __name__ == "__main__":
    main()
