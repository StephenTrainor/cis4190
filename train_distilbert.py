import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_TO_ID = {"FoxNews": 0, "NBC": 1}
ID_TO_LABEL = {0: "FoxNews", 1: "NBC"}


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
                labels.append(LABEL_TO_ID["FoxNews"])
            elif "nbcnews.com" in url:
                labels.append(LABEL_TO_ID["NBC"])
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


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 4
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


def train(config: TrainConfig, train_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = read_rows(train_csv)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = HeadlineDataset(encodings, label_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(loader))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model artifacts to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train headline-only DistilBERT classifier.")
    parser.add_argument("--train_csv", default="url_with_headlines.csv")
    parser.add_argument("--output_dir", default="distilbert_artifacts")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
    )
    train(cfg, Path(args.train_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
