import torch
from typing import Tuple, List, Any

import time
import random
import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urlparse
import re
import os
import html
import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; YourBot/1.0; +https://example.com/bot)"
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def load_csv_to_df(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame and clean headline text when present."""
    df = pd.read_csv(path, encoding=encoding)
    if "h1" in df.columns:
        df["h1"] = df["h1"].map(clean_headline)
    return df


def df_to_csv(df: pd.DataFrame, path: str, index: bool = False, encoding: str = "utf-8") -> None:
    """Save a pandas DataFrame to CSV."""
    df.to_csv(path, index=index, encoding=encoding)


def clean_wrapping_quotes(text: Any) -> str:
    """
    Remove unnecessary matching single/double quotes around a string.
    Example: '"headline"' -> 'headline'
    """
    if text is None:
        return ""
    cleaned = str(text).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def normalize_whitespace(text: Any) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_headline(text: Any) -> str:
    """Basic headline cleaner composed of quote and whitespace normalization."""
    decoded = str(text) if text is not None else ""
    # Some sources are double-encoded (e.g., "&amp;#x27;"), so decode repeatedly.
    for _ in range(3):
        next_decoded = html.unescape(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return normalize_whitespace(clean_wrapping_quotes(decoded))


def drop_rows_missing_required(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Drop rows that are missing any required fields."""
    return df.dropna(subset=required_columns).copy()


def fill_missing_text_with_empty(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """Fill missing values for text columns with empty strings."""
    cleaned_df = df.copy()
    for col in text_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna("")
    return cleaned_df


def preprocess_dataframe(df: pd.DataFrame, headline_column: str = "h1", url_column: str = "url") -> pd.DataFrame:
    """
    Generic preprocessing for crawler/model inputs.
    - Ensures required columns are present
    - Removes rows with missing URLs
    - Cleans headline text
    """
    if url_column not in df.columns:
        raise ValueError(f"Missing required column: {url_column}")

    cleaned_df = drop_rows_missing_required(df, [url_column])
    if headline_column in cleaned_df.columns:
        cleaned_df = fill_missing_text_with_empty(cleaned_df, [headline_column])
        cleaned_df[headline_column] = cleaned_df[headline_column].map(clean_headline)
    return cleaned_df


def deduplicate_by_url(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    """Drop duplicate URLs to avoid repeated training/inference samples."""
    if url_column not in df.columns:
        raise ValueError(f"Missing required column: {url_column}")
    deduped = df.copy()
    deduped[url_column] = deduped[url_column].map(_normalize_url)
    return deduped.drop_duplicates(subset=[url_column], keep="first").copy()

def _normalize_url(url: Any) -> str:
    if url is None:
        return ""
    cleaned = str(url).strip().strip('"').strip("'")
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if not parsed.scheme:
        cleaned = f"https://{cleaned}"
    return cleaned


def _extract_headline(html: str) -> str:
    # Fast-path extraction from common metadata before full DOM parsing.
    og_match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if og_match:
        return og_match.group(1).strip()

    tw_match = re.search(
        r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tw_match:
        return tw_match.group(1).strip()

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
        if title:
            return title

    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1 and h1.get_text(" ", strip=True):
        return h1.get_text(" ", strip=True)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    if tw_title and tw_title.get("content"):
        return tw_title["content"].strip()

    if soup.title and soup.title.get_text(" ", strip=True):
        return soup.title.get_text(" ", strip=True)

    return ""


def fetch_h1(url, max_retries=4, base_delay=2):
    normalized_url = _normalize_url(url)
    if not normalized_url:
        return ""

    for attempt in range(max_retries):
        try:
            r = session.get(normalized_url, timeout=(4, 8), allow_redirects=True)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_for = int(retry_after)
                    except ValueError:
                        sleep_for = base_delay * (2 ** attempt)
                else:
                    sleep_for = base_delay * (2 ** attempt)

                time.sleep(sleep_for + random.uniform(0.5, 1.5))
                continue

            if r.status_code in (403, 404):
                return ""

            r.raise_for_status()
            return _extract_headline(r.text)

        except requests.RequestException:
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0.5, 1.5))

    return ""


def crawl(
    urls,
    min_delay=0.5,
    max_delay=0.29,
    show_progress=False,
    progress_every=25,
):
    results = []
    total = len(urls)
    started_at = time.time()
    found_count = 0

    if total == 0:
        return []

    for idx, url in enumerate(urls, start=1):
        headline = fetch_h1(url)
        normalized_url = _normalize_url(url)
        results.append({"url": normalized_url, "h1": headline})
        if headline:
            found_count += 1

        if show_progress and (idx % progress_every == 0 or idx == total):
            elapsed = max(time.time() - started_at, 1e-9)
            rate = idx / elapsed
            pct = (idx / total) * 100 if total else 100.0
            success_pct = (found_count / idx) * 100 if idx else 0.0
            print(
                f"[crawl] {idx}/{total} ({pct:.1f}%) "
                f"headlines_found={found_count} ({success_pct:.1f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} urls/s"
            )

        # Light jitter to avoid predictable request bursts while staying fast.
        time.sleep(random.uniform(min_delay, max_delay))
    return results


def _read_urls_from_csv(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            normalized_headers = {h.strip().lower(): h for h in reader.fieldnames if h}
            url_header = normalized_headers.get("url")
        else:
            url_header = None

        if url_header:
            for row in reader:
                normalized_url = _normalize_url(row.get(url_header))
                if normalized_url:
                    urls.append(normalized_url)
            return urls

    # Fallback for non-standard CSVs where URLs may appear as first-column values.
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header when present
        for row in reader:
            if not row:
                continue
            normalized_url = _normalize_url(row[0])
            if normalized_url:
                urls.append(normalized_url)
    return urls


def _encode_label_from_url(url: str) -> int:
    normalized = _normalize_url(url).lower()
    domain = urlparse(normalized).netloc
    if "foxnews.com" in domain:
        return 1
    if "nbcnews.com" in domain:
        return 0
    # Unknown publisher
    return -1

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    urls = _read_urls_from_csv(path)
    crawled = crawl(urls)

    # X contains cleaned headline text aligned by row, y is encoded from URL source:
    # FOX NEWS -> 1, NBC -> 0, unknown -> -1.
    X: List[str] = [clean_headline(item["h1"]) for item in crawled]
    y = torch.tensor([_encode_label_from_url(item["url"]) for item in crawled], dtype=torch.long)
    return X, y


if __name__ == "__main__":
    input_csv = "url_only_data.csv"
    output_csv = "headlines_data.csv"

    if os.path.exists(output_csv):
        print(f"Found existing {output_csv}; skipping scraping.")
        crawled_df = load_csv_to_df(output_csv)
    else:
        urls = _read_urls_from_csv(input_csv)
        crawled = crawl(urls, show_progress=True)
        crawled_df = pd.DataFrame(crawled)
    crawled_df = preprocess_dataframe(crawled_df, headline_column="h1", url_column="url")
    crawled_df = deduplicate_by_url(crawled_df, url_column="url")
    df_to_csv(crawled_df, output_csv)
    print(f"Wrote {len(crawled_df)} rows to {output_csv}")

    print("\nDataFrame inspection")
    print(f"shape: {crawled_df.shape}")
    print(f"columns: {list(crawled_df.columns)}")
    print("null counts:")
    print(crawled_df.isna().sum())
    print("\nfirst 10 rows:")
    print(crawled_df.head(10))

