"""
Phase 1 — Spider Data Extraction

Reads the raw Spider dataset files and produces a unified, simplified JSON
for downstream phases. Combines train_spider.json and train_others.json into
a single training split, assigns stable IDs, and extracts SQL operator
patterns for each sample.

Input:
    data/spider/train_spider.json
    data/spider/train_others.json
    data/spider/dev.json

Output:
    data/spider/extracted/train.json
    data/spider/extracted/dev.json

Sample format:
    {
        "id": "spider-train-00001",
        "db_id": "concert_singer",
        "question": "How many singers do we have?",
        "query": "SELECT count(*) FROM singer",
        "sql_patterns": ["COUNT", "FROM", "SELECT"],
        "source": "spider"   // "spider" | "others"
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sql_validator import extract_sql_patterns

RAW_DIR = PROJECT_ROOT / "data" / "spider"
OUT_DIR = PROJECT_ROOT / "data" / "spider" / "extracted"


def load_raw(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_sample(raw: dict, idx: int, split: str, source: str) -> dict:
    return {
        "id": f"spider-{split}-{idx:05d}",
        "db_id": raw["db_id"],
        "question": raw["question"],
        "query": raw["query"],
        "sql_patterns": extract_sql_patterns(raw["query"]),
        "source": source,
    }


def extract_split(samples_raw: list[dict], split: str, source: str) -> list[dict]:
    return [build_sample(r, i + 1, split, source) for i, r in enumerate(samples_raw)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Training split ---
    spider_raw = load_raw(RAW_DIR / "train_spider.json")
    others_raw = load_raw(RAW_DIR / "train_others.json")

    train_spider = extract_split(spider_raw, "train", "spider")
    train_others = extract_split(others_raw, "train", "others")

    # Re-index combined list to avoid ID collisions
    train_all = []
    for idx, s in enumerate(train_spider + train_others, start=1):
        s = dict(s)
        s["id"] = f"spider-train-{idx:05d}"
        train_all.append(s)

    out_train = OUT_DIR / "train.json"
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_all, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_spider)} (spider) + {len(train_others)} (others) = {len(train_all)} samples → {out_train}")

    # --- Dev split ---
    dev_raw = load_raw(RAW_DIR / "dev.json")
    dev = extract_split(dev_raw, "dev", "spider")

    out_dev = OUT_DIR / "dev.json"
    with open(out_dev, "w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False, indent=2)

    print(f"Dev:   {len(dev)} samples → {out_dev}")

    # --- Summary ---
    from collections import Counter
    pattern_counts = Counter("|".join(s["sql_patterns"]) for s in train_all)
    print(f"\nUnique SQL pattern combinations in train: {len(pattern_counts)}")


if __name__ == "__main__":
    main()
