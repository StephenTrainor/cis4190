import pandas as pd

from preprocess import clean_headline, _normalize_url


def _load_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"url", "h1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    standardized = df[["url", "h1"]].copy()
    standardized["url"] = standardized["url"].map(_normalize_url)
    standardized["h1"] = standardized["h1"].map(clean_headline)
    standardized = standardized[standardized["url"] != ""]
    standardized = standardized.drop_duplicates(subset=["url"], keep="first")
    return standardized


def compare_csvs(reference_csv: str, candidate_csv: str, preview_limit: int = 25) -> None:
    ref_df = _load_and_standardize(reference_csv)
    cand_df = _load_and_standardize(candidate_csv)

    ref_urls = set(ref_df["url"])
    cand_urls = set(cand_df["url"])

    only_in_reference = sorted(ref_urls - cand_urls)
    only_in_candidate = sorted(cand_urls - ref_urls)

    merged = ref_df.merge(cand_df, on="url", how="inner", suffixes=("_reference", "_candidate"))
    title_mismatches = merged[merged["h1_reference"] != merged["h1_candidate"]].copy()

    print("=== Comparison Summary ===")
    print(f"Reference rows (deduped by url): {len(ref_df)}")
    print(f"Candidate rows (deduped by url): {len(cand_df)}")
    print(f"Shared urls: {len(merged)}")
    print(f"URLs only in {reference_csv}: {len(only_in_reference)}")
    print(f"URLs only in {candidate_csv}: {len(only_in_candidate)}")
    print(f"Title mismatches on shared urls: {len(title_mismatches)}")

    if only_in_reference:
        print(f"\n=== URLs only in {reference_csv} (first {preview_limit}) ===")
        for url in only_in_reference[:preview_limit]:
            print(url)

    if only_in_candidate:
        print(f"\n=== URLs only in {candidate_csv} (first {preview_limit}) ===")
        for url in only_in_candidate[:preview_limit]:
            print(url)

    if not title_mismatches.empty:
        print(f"\n=== Title mismatches (first {preview_limit}) ===")
        for _, row in title_mismatches.head(preview_limit).iterrows():
            print(f"URL: {row['url']}")
            print(f"  {reference_csv}: {row['h1_reference']}")
            print(f"  {candidate_csv}: {row['h1_candidate']}")
            print()


if __name__ == "__main__":
    compare_csvs("url_with_headlines.csv", "headlines_data.csv")
