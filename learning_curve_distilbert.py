import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple


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


class HeadlineDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_and_eval(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    model_name: str,
    weight_decay: float,
    warmup_ratio: float,
) -> float:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

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


def parse_sizes(raw: str) -> List[int]:
    parsed = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("k"):
            parsed.append(int(float(token[:-1]) * 1000))
        else:
            parsed.append(int(token))
    unique_sorted = sorted(set(x for x in parsed if x > 0))
    if not unique_sorted:
        raise ValueError("No valid positive sizes passed to --sizes.")
    return unique_sorted


def write_results_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = ["train_size", "mean_acc", "std_acc", "min_acc", "max_acc", "repeats"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(results: List[Dict[str, float]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG plot.")
        return

    sizes = [int(r["train_size"]) for r in results]
    means = [r["mean_acc"] for r in results]
    stds = [r["std_acc"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, means, marker="o", label="Mean accuracy")
    plt.fill_between(
        sizes,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning curve: DistilBERT headline classifier")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote plot to {out_png}")


def main() -> None:
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Run learning-curve sweep for DistilBERT.")
    parser.add_argument("--csv_path", default="headlines_data.csv")
    parser.add_argument("--sizes", default="100,200,500,1000,2000")
    parser.add_argument("--repeats", type=int, default=3, help="Random subsets per train size.")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--out_csv", default="learning_curve_results.csv")
    parser.add_argument("--plot_png", default="learning_curve_results.png")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val_frac must be in (0, 1)")

    sizes = parse_sizes(args.sizes)
    texts, labels = read_rows(Path(args.csv_path))
    if len(texts) < 10:
        raise ValueError("Not enough labeled rows parsed from CSV.")

    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts,
        labels,
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=labels,
    )

    max_usable = len(tr_texts)
    usable_sizes = [s for s in sizes if s <= max_usable]
    skipped = [s for s in sizes if s > max_usable]
    if skipped:
        print(f"Skipping sizes larger than training pool ({max_usable}): {skipped}")
    if not usable_sizes:
        raise ValueError(f"All requested sizes exceed training pool ({max_usable}).")

    results: List[Dict[str, float]] = []
    for size in usable_sizes:
        run_scores: List[float] = []
        for run_i in range(args.repeats):
            subset_texts, _, subset_labels, _ = train_test_split(
                tr_texts,
                tr_labels,
                train_size=size,
                random_state=args.seed + run_i,
                stratify=tr_labels,
            )
            acc = train_and_eval(
                train_texts=subset_texts,
                train_labels=subset_labels,
                val_texts=va_texts,
                val_labels=va_labels,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                model_name=args.model_name,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
            )
            run_scores.append(acc)
            print(f"size={size:4d} run={run_i + 1}/{args.repeats} acc={acc:.6f}")

        row = {
            "train_size": float(size),
            "mean_acc": mean(run_scores),
            "std_acc": pstdev(run_scores) if len(run_scores) > 1 else 0.0,
            "min_acc": min(run_scores),
            "max_acc": max(run_scores),
            "repeats": float(args.repeats),
        }
        results.append(row)
        print(
            f"size={size:4d} mean={row['mean_acc']:.6f} std={row['std_acc']:.6f} "
            f"min={row['min_acc']:.6f} max={row['max_acc']:.6f}"
        )

    out_csv = Path(args.out_csv)
    write_results_csv(out_csv, results)
    print(f"Wrote results to {out_csv}")

    if not args.no_plot:
        maybe_plot(results, Path(args.plot_png))


if __name__ == "__main__":
    main()
