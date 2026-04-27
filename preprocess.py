import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 15


def _label_from_url(url: str) -> str:
    """Return 'FoxNews' or 'NBC' based on the URL domain."""
    if "foxnews.com" in url:
        return "FoxNews"
    elif "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unknown source for URL: {url}")


def _scrape_headline(url: str) -> str | None:
    """Fetch the page at `url` and return the <h1> headline text, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
    except Exception:
        pass
    return None


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Read URLs from the CSV at `path`, scrape each headline, and determine
    the news source (FoxNews / NBC) from the URL.

    Returns
    -------
    (X, y) where
        X : list[str]  – scraped headline strings
        y : list[str]  – corresponding labels ("FoxNews" or "NBC")
    """
    df = pd.read_csv(path)
    urls = df["url"].tolist()

    headlines: List[str] = []
    labels: List[str] = []

    with ThreadPoolExecutor(max_workers=16) as pool:
        future_to_url = {pool.submit(_scrape_headline, url): url for url in urls}
        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            headline = future.result()
            if headline is not None:
                headlines.append(headline)
                labels.append(_label_from_url(url))
            if (i + 1) % 200 == 0:
                print(f"  scraped {i + 1}/{len(urls)} URLs …")

    print(f"Done – kept {len(headlines)}/{len(urls)} headlines "
          f"(Fox: {labels.count('FoxNews')}, NBC: {labels.count('NBC')})")
    return headlines, labels


if __name__ == "__main__":
    import csv, os

    csv_in = "url_only_data.csv"
    csv_out = "headlines_data.csv"

    headlines, labels = prepare_data(csv_in)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "source"])
        for h, l in zip(headlines, labels):
            writer.writerow([h, l])

    print(f"Saved {len(headlines)} rows to {csv_out}")