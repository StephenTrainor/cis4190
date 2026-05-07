import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from sklearn.model_selection import train_test_split
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


def word_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len(stripped.split())


def parse_bucket_bounds(raw: str) -> List[int]:
    bounds: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        bounds.append(int(token))
    unique_sorted = sorted(set(bounds))
    if not unique_sorted:
        raise ValueError("No valid values were parsed from --bucket_bounds.")
    if unique_sorted[0] < 0:
        raise ValueError("--bucket_bounds cannot include negative numbers.")
    return unique_sorted


def bucket_name(length: int, bounds: Sequence[int]) -> str:
    previous = 0
    for upper in bounds:
        if length <= upper:
            return f"w_{previous:02d}_{upper:02d}"
        previous = upper + 1
    return f"w_{previous:02d}_plus"


def ordered_bucket_keys(bounds: Sequence[int]) -> List[str]:
    names = [bucket_name(upper, bounds) for upper in bounds]
    names.append(bucket_name(bounds[-1] + 1, bounds))
    return names


def bucket_display_labels(bounds: Sequence[int]) -> List[str]:
    prev = 0
    labels: List[str] = []
    for upper in bounds:
        labels.append(f"{prev}-{upper}")
        prev = upper + 1
    labels.append(f"{prev}+")
    return labels


def maybe_plot_length_buckets(
    bucket_bounds: Sequence[int],
    bucket_stats: Dict[str, Dict[str, float]],
    overall_acc: float,
    out_png: Path,
) -> None:
    try:
        import os

        mpl_dir = Path(__file__).resolve().parent / ".mplcache"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping bucket accuracy plot.")
        return

    keys = ordered_bucket_keys(bucket_bounds)
    x_labels = bucket_display_labels(bucket_bounds)
    accs = [float(bucket_stats[k]["accuracy"]) for k in keys]
    counts = [int(bucket_stats[k]["count"]) for k in keys]

    fig_w = max(7.0, 1.15 * len(keys))
    _, ax = plt.subplots(figsize=(fig_w, 5))
    x_pos = list(range(len(keys)))
    bars = ax.bar(x_pos, accs, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{lab} words" for lab in x_labels], rotation=22, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Headline length (word count)")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(
        overall_acc,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall ({overall_acc:.3f})",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for bar, n in zip(bars, counts):
        ax.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_title("DistilBERT: test accuracy by headline length bucket")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote plot to {out_png}")


def write_bucket_metrics_csv(
    path: Path,
    bucket_bounds: Sequence[int],
    bucket_stats: Dict[str, Dict[str, float]],
    overall_acc: float,
) -> None:
    keys = ordered_bucket_keys(bucket_bounds)
    labels = bucket_display_labels(bucket_bounds)
    fieldnames = ["bucket_key", "word_range", "count", "correct", "accuracy"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, wr in zip(keys, labels):
            stats = bucket_stats[key]
            count = int(stats["count"])
            correct = int(stats["correct"])
            writer.writerow(
                {
                    "bucket_key": key,
                    "word_range": wr,
                    "count": count,
                    "correct": correct,
                    "accuracy": f"{stats['accuracy']:.6f}",
                }
            )
        writer.writerow(
            {
                "bucket_key": "overall",
                "word_range": "",
                "count": "",
                "correct": "",
                "accuracy": f"{overall_acc:.6f}",
            }
        )
    print(f"Wrote metrics to {path}")


def train_model(
    model,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch_i in range(epochs):
        running_loss = 0.0
        step_count = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            step_count += 1
        avg_loss = running_loss / max(step_count, 1)
        print(f"epoch={epoch_i + 1}/{epochs} train_loss={avg_loss:.6f}")


def evaluate_model(
    model,
    test_loader: DataLoader,
    test_word_counts: Sequence[int],
    bucket_bounds: Sequence[int],
    device: torch.device,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"]
            features = {k: v.to(device) for k, v in batch.items()}
            logits = model(**features).logits.cpu()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    if len(all_preds) != len(test_word_counts):
        raise ValueError("Prediction count and test_word_counts are misaligned.")

    total = len(all_labels)
    total_correct = sum(int(p == y) for p, y in zip(all_preds, all_labels))
    overall_acc = total_correct / max(total, 1)

    bucket_names = ordered_bucket_keys(bucket_bounds)
    bucket_stats: Dict[str, Dict[str, float]] = {
        name: {"count": 0.0, "correct": 0.0} for name in bucket_names
    }
    for pred, label, wc in zip(all_preds, all_labels, test_word_counts):
        name = bucket_name(wc, bounds=bucket_bounds)
        bucket_stats[name]["count"] += 1.0
        bucket_stats[name]["correct"] += float(pred == label)

    for name, stats in bucket_stats.items():
        count = int(stats["count"])
        correct = int(stats["correct"])
        stats["accuracy"] = correct / max(count, 1)
        print(
            f"bucket={name:12s} count={count:4d} correct={correct:4d} "
            f"accuracy={stats['accuracy']:.6f}"
        )

    best_bucket = max(bucket_stats.items(), key=lambda kv: kv[1]["accuracy"])[0]
    print(f"overall_test_accuracy={overall_acc:.6f}")
    print(f"best_length_bucket={best_bucket}")
    return overall_acc, bucket_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DistilBERT on headline source classification and test whether "
            "longer headlines are predicted more accurately."
        )
    )
    parser.add_argument("--csv_path", default="headlines_data.csv")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--short_max_words",
        type=int,
        default=7,
        help="Backward-compatible fallback for 3 buckets when --bucket_bounds is not set.",
    )
    parser.add_argument(
        "--medium_max_words",
        type=int,
        default=13,
        help="Backward-compatible fallback for 3 buckets when --bucket_bounds is not set.",
    )
    parser.add_argument(
        "--bucket_bounds",
        default="7,10,13,16",
        help=(
            "Comma-separated inclusive upper bounds for word-count buckets. "
            "Example: 5,8,11,14 creates buckets w_00_05, w_06_08, w_09_11, w_12_14, w_15_plus."
        ),
    )
    parser.add_argument(
        "--plot_png",
        default="length_accuracy_by_bucket.png",
        help="Path to save a bar chart of per-bucket test accuracy.",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the plot.")
    parser.add_argument(
        "--metrics_csv",
        default="",
        help="If set, write per-bucket metrics plus overall accuracy to this CSV path.",
    )
    args = parser.parse_args()

    if not (0.0 < args.test_frac < 1.0):
        raise ValueError("--test_frac must be in (0, 1).")
    if args.short_max_words >= args.medium_max_words:
        raise ValueError("--short_max_words must be < --medium_max_words.")
    if args.bucket_bounds.strip():
        bucket_bounds = parse_bucket_bounds(args.bucket_bounds)
    else:
        # Keep compatibility if someone passes an empty --bucket_bounds string.
        bucket_bounds = [args.short_max_words, args.medium_max_words]

    texts, labels = read_rows(Path(args.csv_path))
    if len(texts) < 20:
        raise ValueError("Not enough labeled rows parsed from CSV.")

    word_counts = [word_count(t) for t in texts]
    indices = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=labels,
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_word_counts = [word_counts[i] for i in test_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    train_ds = HeadlineDataset(train_enc, torch.tensor(train_labels, dtype=torch.long))
    test_ds = HeadlineDataset(test_enc, torch.tensor(test_labels, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    print(f"train_size={len(train_texts)} test_size={len(test_texts)}")
    print(f"length_bucket_bounds={bucket_bounds}")

    train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        device=device,
    )

    overall_acc, bucket_stats = evaluate_model(
        model=model,
        test_loader=test_loader,
        test_word_counts=test_word_counts,
        bucket_bounds=bucket_bounds,
        device=device,
    )

    if args.metrics_csv.strip():
        write_bucket_metrics_csv(
            Path(args.metrics_csv),
            bucket_bounds=bucket_bounds,
            bucket_stats=bucket_stats,
            overall_acc=overall_acc,
        )

    if not args.no_plot:
        maybe_plot_length_buckets(
            bucket_bounds=bucket_bounds,
            bucket_stats=bucket_stats,
            overall_acc=overall_acc,
            out_png=Path(args.plot_png),
        )


if __name__ == "__main__":
    main()
