"""
Phase 1 — BIRD Data Extraction

Reads the raw BIRD dataset files and produces a unified, simplified JSON
for downstream phases. Assigns stable IDs and preserves all BIRD-specific
fields, including the `evidence` field required for downstream evaluation.

Input:
    data/bird/train/train.json
    data/bird/dev/dev.json

Output:
    data/bird/extracted/train.json
    data/bird/extracted/dev.json

Sample format (train):
    {
        "id": "bird-train-00001",
        "db_id": "movie_platform",
        "question": "Name movie titles released in year 1945...",
        "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
        "SQL": "SELECT movie_title FROM movies WHERE ...",
        "sql_patterns": ["FROM", "SELECT", "WHERE"],
        "difficulty": null    // null for train; "simple"|"moderate"|"challenging" for dev
    }

Note: `evidence` may be an empty string for a minority of samples (~7% train, ~10% dev).
Both `question` and `evidence` are translated together in Phase 3 to maintain
terminological consistency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sql_validator import extract_sql_patterns

RAW_DIR = PROJECT_ROOT / "data" / "bird"
OUT_DIR = PROJECT_ROOT / "data" / "bird" / "extracted"


def load_raw(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_sample(raw: dict, idx: int, split: str) -> dict:
    return {
        "id": f"bird-{split}-{idx:05d}",
        "db_id": raw["db_id"],
        "question": raw["question"],
        "evidence": raw.get("evidence", ""),
        "SQL": raw["SQL"],
        "sql_patterns": extract_sql_patterns(raw["SQL"]),
        "difficulty": raw.get("difficulty", None),  # present in dev, absent in train
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Training split ---
    train_raw = load_raw(RAW_DIR / "train" / "train.json")
    train = [build_sample(r, i + 1, "train") for i, r in enumerate(train_raw)]

    out_train = OUT_DIR / "train.json"
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    has_evidence = sum(1 for s in train if s["evidence"].strip())
    print(f"Train: {len(train)} samples → {out_train}")
    print(f"  Non-empty evidence: {has_evidence}/{len(train)} ({has_evidence/len(train)*100:.1f}%)")

    # --- Dev split ---
    dev_raw = load_raw(RAW_DIR / "dev" / "dev.json")
    dev = [build_sample(r, i + 1, "dev") for i, r in enumerate(dev_raw)]

    out_dev = OUT_DIR / "dev.json"
    with open(out_dev, "w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False, indent=2)

    from collections import Counter
    diff_counts = Counter(s["difficulty"] for s in dev)
    has_evidence_dev = sum(1 for s in dev if s["evidence"].strip())
    print(f"\nDev: {len(dev)} samples → {out_dev}")
    print(f"  Difficulty distribution: {dict(diff_counts)}")
    print(f"  Non-empty evidence: {has_evidence_dev}/{len(dev)} ({has_evidence_dev/len(dev)*100:.1f}%)")

    # --- Summary ---
    from collections import Counter as C
    pattern_counts = C("|".join(s["sql_patterns"]) for s in train)
    print(f"\nUnique SQL pattern combinations in train: {len(pattern_counts)}")


if __name__ == "__main__":
    main()
