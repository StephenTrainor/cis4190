import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model.pt from local transformer artifacts.")
    parser.add_argument("--artifacts_dir", default="submission_roberta_artifacts")
    parser.add_argument("--output", default="model.pt")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Missing artifacts directory: {artifacts_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(
        artifacts_dir,
        use_safetensors=True,
    )
    bundle = dict(model.state_dict())
    bundle["__bundle_type__"] = "hf_roberta_single"
    bundle["hf_config"] = model.config.to_dict()
    bundle["id_to_label"] = {0: "FoxNews", 1: "NBC"}

    tokenizer_files = {}
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        file_path = artifacts_dir / filename
        if file_path.exists():
            tokenizer_files[filename] = file_path.read_bytes()
    if not tokenizer_files:
        raise FileNotFoundError("No tokenizer files found in artifacts directory.")
    bundle["tokenizer_files"] = tokenizer_files
    torch.save(bundle, args.output)
    print(f"Saved transformer state_dict to: {args.output}")


if __name__ == "__main__":
    main()
