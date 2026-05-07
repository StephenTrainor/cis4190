import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_learning_curve_csv(path: Path) -> Tuple[List[int], List[float], List[float]]:
    sizes: List[int] = []
    means: List[float] = []
    stds: List[float] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"train_size", "mean_acc", "std_acc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            size = int(float(row["train_size"]))
            mean_acc = float(row["mean_acc"])
            std_acc = float(row["std_acc"])
            sizes.append(size)
            means.append(mean_acc)
            stds.append(std_acc)

    if not sizes:
        raise ValueError("No rows found in the input CSV.")
    return sizes, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curve from CSV results.")
    parser.add_argument("--csv_path", default="learning_curve_results.csv")
    parser.add_argument("--out_png", default="learning_curve_plot.png")
    parser.add_argument("--title", default="Learning Curve: Accuracy vs Training Size")
    parser.add_argument("--show", action="store_true", help="Display plot window after saving.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_png = Path(args.out_png)
    sizes, means, stds = read_learning_curve_csv(csv_path)

    lower = [max(0.0, m - s) for m, s in zip(means, stds)]
    upper = [min(1.0, m + s) for m, s in zip(means, stds)]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, means, marker="o", linewidth=2, label="Mean accuracy")
    plt.fill_between(sizes, lower, upper, alpha=0.2, label="±1 std")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote plot to {out_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
