#!/usr/bin/env python3
"""
Analyze linguistic differences between FOX and NBC headlines.

Usage:
  ./venv/bin/python analyze_fox_vs_nbc.py --input headlines_data.csv
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import pandas as pd


TOKEN_RE = re.compile(r"[a-z']+")

# Compact but practical stopword list for headline analysis.
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "s",
}

# Lightweight lexicons for headline-level polarity and style.
POSITIVE_WORDS = {
    "success", "wins", "win", "improves", "improve", "growth", "gains", "gain",
    "supports", "support", "hope", "safe", "peace", "good", "better", "best",
    "strong", "strength", "help", "helps", "helped", "saved", "save", "boost",
}
NEGATIVE_WORDS = {
    "attack", "attacks", "killed", "kill", "killing", "war", "crisis", "fear",
    "fears", "angry", "rage", "violent", "violence", "threat", "threats",
    "crime", "dead", "death", "dies", "injured", "chaos", "scandal", "blasts",
    "outrage", "risk", "danger", "dangerous", "shocking",
}
EMOTIONAL_WORDS = {
    "outrage", "shocking", "furious", "rage", "fear", "fearmongering", "crisis",
    "chaos", "angry", "panic", "dramatic", "bombshell", "blasts", "slams",
    "stunning", "surprise",
}
REPORTING_TONE_WORDS = {
    "reports", "report", "says", "according", "official", "officials", "analysis",
    "data", "study", "states", "statement", "announces", "announced", "update",
    "updates", "confirms", "confirmed", "sources",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FOX vs NBC headline linguistic analysis.")
    parser.add_argument("--input", default="headlines_data.csv", help="Input CSV path.")
    parser.add_argument(
        "--output",
        default="fox_vs_nbc_analysis.md",
        help="Output markdown report path.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Number of top words per section.")
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=8,
        help="Minimum total count before a word is included in comparative ranking.",
    )
    return parser.parse_args()


def infer_source(url: str) -> str | None:
    netloc = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in netloc:
        return "FOX"
    if "nbcnews.com" in netloc or "bncnews.com" in netloc:
        # Treat user-mentioned "BNC" as NBC/BNC bucket.
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


def top_words(counter: Counter, n: int) -> List[Tuple[str, int]]:
    return counter.most_common(n)


def comparative_keywords(
    fox_counts: Counter,
    nbc_counts: Counter,
    min_word_count: int,
    n: int,
) -> Tuple[List[Tuple[str, float, int, int]], List[Tuple[str, float, int, int]]]:
    fox_total = sum(fox_counts.values())
    nbc_total = sum(nbc_counts.values())
    vocab = set(fox_counts) | set(nbc_counts)
    vocab_size = len(vocab)

    fox_skew: List[Tuple[str, float, int, int]] = []
    nbc_skew: List[Tuple[str, float, int, int]] = []

    for word in vocab:
        fox_c = fox_counts.get(word, 0)
        nbc_c = nbc_counts.get(word, 0)
        if fox_c + nbc_c < min_word_count:
            continue

        # Smoothed log-odds ratio: >0 means FOX-skewed, <0 means NBC-skewed.
        fox_rate = (fox_c + 1) / (fox_total + vocab_size)
        nbc_rate = (nbc_c + 1) / (nbc_total + vocab_size)
        log_odds = math.log(fox_rate / nbc_rate)

        item = (word, log_odds, fox_c, nbc_c)
        if log_odds >= 0:
            fox_skew.append(item)
        else:
            nbc_skew.append(item)

    fox_skew.sort(key=lambda x: x[1], reverse=True)
    nbc_skew.sort(key=lambda x: x[1])
    return fox_skew[:n], nbc_skew[:n]


def lexicon_metrics(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(1 for token in tokens if token in lexicon)


def sentiment_and_style_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}

    for source in ("FOX", "NBC"):
        source_df = df[df["source"] == source]
        all_tokens = [token for h in source_df["headline"] for token in tokenize(h)]
        total_tokens = max(len(all_tokens), 1)

        positive = lexicon_metrics(all_tokens, POSITIVE_WORDS)
        negative = lexicon_metrics(all_tokens, NEGATIVE_WORDS)
        emotional = lexicon_metrics(all_tokens, EMOTIONAL_WORDS)
        reporting = lexicon_metrics(all_tokens, REPORTING_TONE_WORDS)

        # Per-1k words makes differences easier to compare across corpora.
        per_1k = 1000.0 / total_tokens
        summary[source] = {
            "headline_count": float(len(source_df)),
            "total_tokens": float(total_tokens),
            "positive_per_1k": positive * per_1k,
            "negative_per_1k": negative * per_1k,
            "sentiment_balance_per_1k": (positive - negative) * per_1k,
            "emotional_per_1k": emotional * per_1k,
            "reporting_per_1k": reporting * per_1k,
        }

    return summary


def format_word_table(rows: List[Tuple[str, int]], headers: Tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    for word, count in rows:
        lines.append(f"| {word} | {count} |")
    return "\n".join(lines)


def format_compare_table(rows: List[Tuple[str, float, int, int]], label: str) -> str:
    lines = [
        f"### {label}",
        "",
        "| Word | log-odds | FOX count | NBC count |",
        "|---|---:|---:|---:|",
    ]
    for word, log_odds, fox_c, nbc_c in rows:
        lines.append(f"| {word} | {log_odds:.3f} | {fox_c} | {nbc_c} |")
    return "\n".join(lines)


def hypothesis_notes(style_summary: Dict[str, Dict[str, float]]) -> List[str]:
    fox = style_summary["FOX"]
    nbc = style_summary["NBC"]

    findings = []
    if fox["emotional_per_1k"] > nbc["emotional_per_1k"]:
        findings.append(
            "- FOX uses more emotional lexicon per 1,000 words than NBC in this dataset."
        )
    else:
        findings.append(
            "- NBC uses equal or more emotional lexicon per 1,000 words than FOX in this dataset."
        )

    if nbc["reporting_per_1k"] > fox["reporting_per_1k"]:
        findings.append(
            "- NBC shows a more reporting/neutral lexical tone than FOX based on the reporting lexicon."
        )
    else:
        findings.append(
            "- FOX shows equal or more reporting/neutral lexical tone than NBC based on the reporting lexicon."
        )

    if fox["negative_per_1k"] > nbc["negative_per_1k"]:
        findings.append("- FOX headlines contain more negative lexicon than NBC on a per-1k basis.")
    else:
        findings.append("- NBC headlines contain equal or more negative lexicon than FOX on a per-1k basis.")

    return findings


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    required_cols = {"url", "headline"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean_df = df.dropna(subset=["url", "headline"]).copy()
    clean_df["source"] = clean_df["url"].map(infer_source)
    clean_df = clean_df[clean_df["source"].isin(["FOX", "NBC"])].copy()
    clean_df["headline"] = clean_df["headline"].astype(str).str.strip()
    clean_df = clean_df[clean_df["headline"] != ""]

    fox_counts = count_words(clean_df.loc[clean_df["source"] == "FOX", "headline"])
    nbc_counts = count_words(clean_df.loc[clean_df["source"] == "NBC", "headline"])

    fox_top = top_words(fox_counts, args.top_n)
    nbc_top = top_words(nbc_counts, args.top_n)
    fox_skew, nbc_skew = comparative_keywords(
        fox_counts,
        nbc_counts,
        min_word_count=args.min_word_count,
        n=args.top_n,
    )
    style_summary = sentiment_and_style_summary(clean_df)

    lines = [
        "# FOX vs NBC Headline Analysis",
        "",
        "## Dataset Scope",
        f"- Input file: `{input_path}`",
        f"- Total rows used: {len(clean_df)}",
        f"- FOX headlines: {int((clean_df['source'] == 'FOX').sum())}",
        f"- NBC headlines: {int((clean_df['source'] == 'NBC').sum())}",
        "",
        "## Top Words by Source",
        "",
        "### FOX top words",
        format_word_table(fox_top, ("Word", "Count")),
        "",
        "### NBC top words",
        format_word_table(nbc_top, ("Word", "Count")),
        "",
        "## Word Frequency Comparison",
        "",
        "Words ranked by smoothed log-odds ratio (higher positive means more FOX-skewed; lower negative means more NBC-skewed).",
        "",
        format_compare_table(fox_skew, "Most FOX-skewed words"),
        "",
        format_compare_table(nbc_skew, "Most NBC-skewed words"),
        "",
        "## Sentiment and Tone Analysis",
        "",
        "| Source | Positive /1k | Negative /1k | Sentiment Balance /1k | Emotional /1k | Reporting /1k |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for source in ("FOX", "NBC"):
        metrics = style_summary[source]
        lines.append(
            f"| {source} | {metrics['positive_per_1k']:.2f} | {metrics['negative_per_1k']:.2f} "
            f"| {metrics['sentiment_balance_per_1k']:.2f} | {metrics['emotional_per_1k']:.2f} "
            f"| {metrics['reporting_per_1k']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Hypothesis Check",
            *hypothesis_notes(style_summary),
            "",
            "## Notes",
            "- This is a lexicon-based analysis (fast and interpretable) rather than a deep contextual sentiment model.",
            "- You can tune or expand the emotion/reporting lexicons in this script to align with your class framing.",
        ]
    )

    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")

    print(f"Saved report to {output_path}")
    print(f"Rows used: {len(clean_df)}")
    print(f"FOX headlines: {(clean_df['source'] == 'FOX').sum()}")
    print(f"NBC headlines: {(clean_df['source'] == 'NBC').sum()}")


if __name__ == "__main__":
    main()
