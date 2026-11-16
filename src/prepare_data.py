
from pathlib import Path
import json
import hashlib
import pandas as pd
from features import clean_data

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bank-additional-full.csv"
BASE_PATH = ROOT / "data" / "base" / "base.csv"
PROCESSED = ROOT / "data" / "processed"

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    raw = pd.read_csv(RAW_PATH, sep=";")
    base = raw.iloc[:int(len(raw) * 0.60)].copy()
    base.to_csv(BASE_PATH, sep=";", index=False)

    clean = clean_data(base)
    y = clean["y"].astype("int8")
    X = clean.drop(columns=["y", "duration"])

    split_index = int(len(clean) * 0.80)
    train = clean.iloc[:split_index].copy()
    test = clean.iloc[split_index:].copy()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", sep=";", index=False)
    test.to_csv(PROCESSED / "test_reserved_for_pr4.csv", sep=";", index=False)

    metadata = {
        "raw_file": "data/raw/bank-additional-full.csv",
        "raw_rows": int(len(raw)),
        "raw_sha256": sha256(RAW_PATH),
        "base_rows_before_deduplication": int(len(base)),
        "base_duplicates_removed": int(base.duplicated().sum()),
        "base_rows_after_deduplication": int(len(clean)),
        "train_rows": int(len(train)),
        "test_rows_reserved_for_pr4": int(len(test)),
        "split_method": "chronological; no shuffle",
        "train_share": 0.80,
        "random_state": 42,
        "target_mapping": {"no": 0, "yes": 1},
        "excluded_features": ["duration", "y"]
    }
    with open(PROCESSED / "split_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
