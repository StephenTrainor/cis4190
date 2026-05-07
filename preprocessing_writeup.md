# Preprocessing Pipeline

## 1) Inputs and entrypoints

There are two practical entrypoints:

- **`prepare_data(path)`**: used to build model inputs/labels (`X`, `y`) directly from a CSV.
- **Script mode (`python preprocess.py`)**: crawls URLs and writes/loads `headlines_data.csv`, then runs DataFrame-level cleaning and URL deduplication.

The file expects URL data to come from CSV rows (primarily a `url` column).

## 2) URL ingestion from CSV

`prepare_data(path)` starts by calling `_read_urls_from_csv(path)`.

`_read_urls_from_csv` behavior:

1. Opens CSV with `csv.DictReader`.
2. Normalizes header names to lowercase/trimmed and looks for a `url` header.
3. If found, reads each row value from that header.
4. Every URL is passed through `_normalize_url(...)`:
   - Converts to string, trims whitespace and wrapping quotes.
   - If scheme is missing, prefixes `https://`.
   - Returns empty string for blank/invalid-ish input.
5. Keeps only non-empty normalized URLs.
6. If no usable `url` header exists, falls back to `csv.reader` and treats first column as URL (skipping one header row).

Result: a list of normalized URL strings.

## 3) Crawling and headline extraction

After URL ingestion, `prepare_data` calls:

- `crawl(urls)`

`crawl` loops through URLs and for each URL:

1. Calls `fetch_h1(url)`.
2. Stores result row as `{"url": normalized_url, "h1": headline}`.

### `fetch_h1(url, max_retries=4, base_delay=2)`

- Uses a shared `requests.Session` with a custom `User-Agent`.
- Normalizes URL first; returns `""` immediately if empty.
- Tries up to 4 attempts.
- Request config: `timeout=(4, 8)`, `allow_redirects=True`.

Response handling:

- **429**: respects `Retry-After` if parseable, otherwise exponential backoff (`base_delay * 2^attempt`) plus random jitter.
- **403 or 404**: returns `""` (no headline).
- Other 2xx/valid responses: extracts headline via `_extract_headline(r.text)`.
- Request exceptions: exponential backoff + jitter, then retry.
- If all attempts fail: returns `""`.

### `_extract_headline(html)`

Headline extraction order:

1. Regex for `<meta property="og:title" ... content="...">`.
2. Regex for `<meta name="twitter:title" ... content="...">`.
3. Regex for `<title>...</title>`.
4. BeautifulSoup fallback:
   - First `<h1>` text
   - `meta[property="og:title"]`
   - `meta[name="twitter:title"]`
   - `soup.title`
5. If none found: `""`.

### Crawl pacing and progress

- `crawl` applies a sleep each loop using `random.uniform(min_delay, max_delay)`.
- Current defaults are `min_delay=0.5` and `max_delay=0.29` (reversed range; Python still samples between them).
- Optional progress logging exists (`show_progress=True`), used in script mode.

## 4) Text cleaning used for headlines

In `prepare_data`, each crawled headline is cleaned with `clean_headline`.

`clean_headline(text)`:

1. Converts `None` to empty string.
2. Repeatedly runs `html.unescape(...)` up to 3 times (handles double-encoded entities like `&amp;#x27;`).
3. Removes matching wrapping quotes via `clean_wrapping_quotes`.
4. Collapses repeated whitespace and trims via `normalize_whitespace`.

Final cleaned headline is the feature text.

## 5) Label generation (`y`)

`prepare_data` derives labels from URL domain via `_encode_label_from_url(url)`:

- Domain containing `foxnews.com` -> `1`
- Domain containing `nbcnews.com` -> `0`
- Any other domain -> `-1` (unknown publisher)

Unlike the older commented-out path in the same function, the active code does **not** drop unknowns; it keeps them as `-1`.

## 6) Final outputs from `prepare_data`

The active implementation returns:

- **`X`**: `List[str]` of cleaned headlines, one per crawled URL.
- **`y`**: `torch.tensor([...], dtype=torch.long)` with integer class codes aligned by index with `X`.

Construction:

- `X = [clean_headline(item["h1"]) for item in crawled]`
- `y = torch.tensor([_encode_label_from_url(item["url"]) for item in crawled], dtype=torch.long)`

So `X[i]` and `y[i]` always correspond to the same crawled row.

## 7) Script mode (`python preprocess.py`) side-path

When running `preprocess.py` directly:

1. Looks for `headlines_data.csv`.
2. If file exists:
   - Loads it with `load_csv_to_df` and cleans `h1` column.
3. Else:
   - Reads URLs from `url_only_data.csv`.
   - Crawls with progress.
   - Builds DataFrame from crawl output.
4. Applies `preprocess_dataframe`:
   - Requires `url` column.
   - Drops rows missing `url`.
   - Fills missing headline text with `""` and cleans it.
5. Applies `deduplicate_by_url`:
   - Normalizes URL strings and drops duplicate URLs (keeps first).
6. Writes result to `headlines_data.csv`.
7. Prints shape/columns/null counts/first rows.

## 8) Practical implications

- If crawling fails for a URL, `X` gets `""` for that row.
- Unknown-source URLs are currently retained in `y` as `-1`.
- `prepare_data` does not perform DataFrame deduplication; duplicates in input URLs can remain in `X`/`y`.
- URL normalization is applied consistently both when reading and when labeling.
