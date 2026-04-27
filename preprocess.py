import csv
import re
from typing import List, Tuple


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _pick_first(row: dict, candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def _extract_label_from_url(url: str) -> str:
    url_l = (url or "").lower()
    if "foxnews.com" in url_l:
        return "FoxNews"
    if "nbcnews.com" in url_l:
        return "NBC"
    return "Unknown"


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Read CSV data and return model-ready text inputs and aligned labels.
    """
    X: List[str] = []
    y: List[str] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = _pick_first(row, ["url", "\ufeffurl", "URL", "Url"])
            headline = _pick_first(row, ["headline", "scraped_headline", "alternative_headline", "title"])

            label = _extract_label_from_url(url)
            if label == "Unknown":
                # Skip rows that do not belong to either target class.
                continue

            headline_norm = _normalize_text(headline)

            # Strict headline-only features.
            features = headline_norm
            X.append(features)
            y.append(label)

    if len(X) != len(y):
        raise ValueError("prepare_data produced misaligned X and y lengths.")
    if not X:
        raise ValueError("prepare_data found no valid rows for FoxNews/NBC.")

    return X, y