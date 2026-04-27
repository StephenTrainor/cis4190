import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_metrics(stdout: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("cv_mean="):
            metrics["cv_mean"] = float(line.split("=", 1)[1])
        elif line.startswith("cv_std="):
            metrics["cv_std"] = float(line.split("=", 1)[1])
    return metrics


def run_one(config: Dict[str, str], workdir: Path) -> Dict[str, str]:
    cmd: List[str] = [
        sys.executable,
        "eval_distilbert_cv.py",
        "--csv_path",
        config["csv_path"],
        "--folds",
        config["folds"],
        "--epochs",
        config["epochs"],
        "--batch_size",
        config["batch_size"],
        "--lr",
        config["lr"],
        "--max_length",
        config["max_length"],
        "--model_name",
        config["model_name"],
        "--weight_decay",
        config["weight_decay"],
        "--warmup_ratio",
        config["warmup_ratio"],
    ]

    print("\nRUN:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
        timeout=int(config.get("timeout_sec", "3600")),
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Run failed with code {proc.returncode}")

    metrics = parse_metrics(proc.stdout)
    out = dict(config)
    out["cv_mean"] = f"{metrics.get('cv_mean', float('nan')):.6f}"
    out["cv_std"] = f"{metrics.get('cv_std', float('nan')):.6f}"
    return out


def default_configs(csv_path: str) -> List[Dict[str, str]]:
    return [
        {
            "name": "distilbert_baseline",
            "csv_path": csv_path,
            "folds": "3",
            "epochs": "3",
            "batch_size": "16",
            "lr": "2e-5",
            "max_length": "64",
            "model_name": "distilbert-base-uncased",
            "weight_decay": "0.01",
            "warmup_ratio": "0.1",
            "timeout_sec": "3600",
        },
        {
            "name": "roberta_best_so_far",
            "csv_path": csv_path,
            "folds": "3",
            "epochs": "3",
            "batch_size": "8",
            "lr": "2e-5",
            "max_length": "96",
            "model_name": "roberta-base",
            "weight_decay": "0.01",
            "warmup_ratio": "0.1",
            "timeout_sec": "5400",
        },
        {
            "name": "roberta_more_epochs",
            "csv_path": csv_path,
            "folds": "3",
            "epochs": "4",
            "batch_size": "8",
            "lr": "2e-5",
            "max_length": "96",
            "model_name": "roberta-base",
            "weight_decay": "0.01",
            "warmup_ratio": "0.1",
            "timeout_sec": "7200",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transformer CV sweep and save CSV results.")
    parser.add_argument("--csv_path", default="url_with_headlines.csv")
    parser.add_argument("--output_csv", default="transformer_sweep_results.csv")
    parser.add_argument("--include_5fold", action="store_true")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    configs = default_configs(args.csv_path)
    if args.include_5fold:
        configs.append(
            {
                "name": "roberta_5fold_check",
                "csv_path": args.csv_path,
                "folds": "5",
                "epochs": "3",
                "batch_size": "8",
                "lr": "2e-5",
                "max_length": "96",
                "model_name": "roberta-base",
                "weight_decay": "0.01",
                "warmup_ratio": "0.1",
                "timeout_sec": "10800",
            }
        )
    results: List[Dict[str, str]] = []
    for cfg in configs:
        results.append(run_one(cfg, workdir))

    output_path = workdir / args.output_csv
    fields = [
        "name",
        "model_name",
        "folds",
        "epochs",
        "batch_size",
        "lr",
        "max_length",
        "weight_decay",
        "warmup_ratio",
        "cv_mean",
        "cv_std",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"\nSaved sweep results to: {output_path}")


if __name__ == "__main__":
    main()
