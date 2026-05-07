#!/usr/bin/env python3
"""
Create presentation-friendly visualizations for FOX vs NBC headline analysis.

Usage:
  ./venv/bin/python visualize_fox_vs_nbc.py --input headlines_data.csv --outdir fox_vs_nbc_viz
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd


TOKEN_RE = re.compile(r"[a-z']+")

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves", "s",
}

POSITIVE_WORDS = {
    "success", "wins", "win", "improves", "improve", "growth", "gains", "gain", "supports",
    "support", "hope", "safe", "peace", "good", "better", "best", "strong", "strength",
    "help", "helps", "helped", "saved", "save", "boost",
}
NEGATIVE_WORDS = {
    "attack", "attacks", "killed", "kill", "killing", "war", "crisis", "fear", "fears",
    "angry", "rage", "violent", "violence", "threat", "threats", "crime", "dead", "death",
    "dies", "injured", "chaos", "scandal", "blasts", "outrage", "risk", "danger",
    "dangerous", "shocking",
}
EMOTIONAL_WORDS = {
    "outrage", "shocking", "furious", "rage", "fear", "fearmongering", "crisis", "chaos",
    "angry", "panic", "dramatic", "bombshell", "blasts", "slams", "stunning", "surprise",
}
REPORTING_TONE_WORDS = {
    "reports", "report", "says", "according", "official", "officials", "analysis", "data",
    "study", "states", "statement", "announces", "announced", "update", "updates",
    "confirms", "confirmed", "sources",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FOX vs NBC visualization assets.")
    parser.add_argument("--input", default="headlines_data.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="fox_vs_nbc_viz", help="Output directory for PNGs")
    parser.add_argument("--top-n", type=int, default=15, help="Words to show in top-words chart")
    parser.add_argument("--min-word-count", type=int, default=8, help="Min total count for skew chart")
    return parser.parse_args()


def infer_source(url: str) -> str | None:
    netloc = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in netloc:
        return "FOX"
    if "nbcnews.com" in netloc or "bncnews.com" in netloc:
        return "NBC"
    return None


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def count_words(headlines: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for headline in headlines:
        counts.update(tokenize(headline))
    return counts


def comparative_keywords(
    fox_counts: Counter,
    nbc_counts: Counter,
    min_word_count: int,
    n: int,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    fox_total = sum(fox_counts.values())
    nbc_total = sum(nbc_counts.values())
    vocab = set(fox_counts) | set(nbc_counts)
    vocab_size = len(vocab)

    fox_skew: List[Tuple[str, float]] = []
    nbc_skew: List[Tuple[str, float]] = []

    for word in vocab:
        fox_c = fox_counts.get(word, 0)
        nbc_c = nbc_counts.get(word, 0)
        if fox_c + nbc_c < min_word_count:
            continue

        fox_rate = (fox_c + 1) / (fox_total + vocab_size)
        nbc_rate = (nbc_c + 1) / (nbc_total + vocab_size)
        log_odds = math.log(fox_rate / nbc_rate)
        if log_odds >= 0:
            fox_skew.append((word, log_odds))
        else:
            nbc_skew.append((word, log_odds))

    fox_skew.sort(key=lambda x: x[1], reverse=True)
    nbc_skew.sort(key=lambda x: x[1])
    return fox_skew[:n], nbc_skew[:n]


def lexicon_count(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(1 for tok in tokens if tok in lexicon)


def sentiment_metrics(tokens: List[str]) -> dict:
    total = max(len(tokens), 1)
    scale = 1000.0 / total
    pos = lexicon_count(tokens, POSITIVE_WORDS)
    neg = lexicon_count(tokens, NEGATIVE_WORDS)
    emo = lexicon_count(tokens, EMOTIONAL_WORDS)
    rep = lexicon_count(tokens, REPORTING_TONE_WORDS)
    return {
        "Positive": pos * scale,
        "Negative": neg * scale,
        "Balance": (pos - neg) * scale,
        "Emotional": emo * scale,
        "Reporting": rep * scale,
    }


def save_top_words_plot(fox_counts: Counter, nbc_counts: Counter, top_n: int, out_png: Path) -> None:
    fox_top = fox_counts.most_common(top_n)
    nbc_top = nbc_counts.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False)

    fox_words = [w for w, _ in reversed(fox_top)]
    fox_vals = [v for _, v in reversed(fox_top)]
    axes[0].barh(fox_words, fox_vals, color="#d62728")
    axes[0].set_title("FOX Top Words")
    axes[0].set_xlabel("Count")

    nbc_words = [w for w, _ in reversed(nbc_top)]
    nbc_vals = [v for _, v in reversed(nbc_top)]
    axes[1].barh(nbc_words, nbc_vals, color="#1f77b4")
    axes[1].set_title("NBC Top Words")
    axes[1].set_xlabel("Count")

    fig.suptitle(f"Top {top_n} Words by Source", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_skew_plot(fox_skew: List[Tuple[str, float]], nbc_skew: List[Tuple[str, float]], out_png: Path) -> None:
    nbc_labels = [w for w, _ in nbc_skew]
    nbc_vals = [v for _, v in nbc_skew]
    fox_labels = [w for w, _ in fox_skew]
    fox_vals = [v for _, v in fox_skew]

    labels = nbc_labels + fox_labels
    values = nbc_vals + fox_vals
    colors = ["#1f77b4"] * len(nbc_labels) + ["#d62728"] * len(fox_labels)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Smoothed log-odds (FOX vs NBC)")
    ax.set_title("Most Source-Skewed Words")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_tone_plot(fox_metrics: dict, nbc_metrics: dict, out_png: Path) -> None:
    categories = ["Positive", "Negative", "Balance", "Emotional", "Reporting"]
    fox_vals = [fox_metrics[c] for c in categories]
    nbc_vals = [nbc_metrics[c] for c in categories]

    x = list(range(len(categories)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], fox_vals, width=width, label="FOX", color="#d62728")
    ax.bar([i + width / 2 for i in x], nbc_vals, width=width, label="NBC", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Rate per 1,000 tokens")
    ax.set_title("Sentiment and Tone Lexicon Comparison")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_source_mix_plot(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["source"].value_counts().reindex(["FOX", "NBC"]).fillna(0).astype(int)
    labels = list(counts.index)
    values = list(counts.values)
    colors = ["#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=120,
        colors=colors,
        textprops={"color": "black"},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Dataset Source Composition")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"url", "headline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["url", "headline"]).copy()
    df["source"] = df["url"].map(infer_source)
    df = df[df["source"].isin(["FOX", "NBC"])].copy()
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"] != ""]

    fox_headlines = df.loc[df["source"] == "FOX", "headline"]
    nbc_headlines = df.loc[df["source"] == "NBC", "headline"]

    fox_counts = count_words(fox_headlines)
    nbc_counts = count_words(nbc_headlines)
    fox_skew, nbc_skew = comparative_keywords(
        fox_counts, nbc_counts, min_word_count=args.min_word_count, n=args.top_n
    )

    fox_tokens = [tok for h in fox_headlines for tok in tokenize(h)]
    nbc_tokens = [tok for h in nbc_headlines for tok in tokenize(h)]
    fox_metrics = sentiment_metrics(fox_tokens)
    nbc_metrics = sentiment_metrics(nbc_tokens)

    save_top_words_plot(fox_counts, nbc_counts, args.top_n, outdir / "top_words_by_source.png")
    save_skew_plot(fox_skew, nbc_skew, outdir / "source_skewed_words.png")
    save_tone_plot(fox_metrics, nbc_metrics, outdir / "sentiment_tone_comparison.png")
    save_source_mix_plot(df, outdir / "dataset_source_mix.png")

    print(f"Wrote visualization assets to {outdir}")
    print(f"- {outdir / 'top_words_by_source.png'}")
    print(f"- {outdir / 'source_skewed_words.png'}")
    print(f"- {outdir / 'sentiment_tone_comparison.png'}")
    print(f"- {outdir / 'dataset_source_mix.png'}")


if __name__ == "__main__":
    main()
