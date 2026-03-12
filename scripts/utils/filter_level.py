"""
Filter a translated L5 file down to any level Lk using pre-built ID lists.

Workflow:
  1. Phase 2: human annotates (or GPT translates) ALL samples in L5_greedy.json
     → produces  data/{dataset}/manual/L5_vi.json  (2,165 Spider / 2,357 BIRD)
  2. This script: given that file + a target level k, outputs L{k}_vi.json
     containing only the IDs that belong to L1..Lk.

Because L1 ⊂ L2 ⊂ L3 ⊂ L4 ⊂ L5 (verified), you never re-translate anything.
Each Lk is just a subset of the already-translated L5.

Usage:
    python scripts/utils/filter_level.py \\
        --dataset spider \\
        --input data/spider/manual/L5_vi.json \\
        --level 3 \\
        --output data/spider/manual/L3_vi.json

    # Or batch-produce all levels at once:
    python scripts/utils/filter_level.py \\
        --dataset spider \\
        --input data/spider/manual/L5_vi.json \\
        --all-levels \\
        --output-dir data/spider/manual/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_level_ids(dataset: str, level: int, strategy: str = "greedy") -> set[str]:
    """Return the set of IDs for Lk (from the pre-built seeds directory)."""
    seeds_dir = PROJECT_ROOT / "data" / dataset / "extracted" / "seeds"
    path = seeds_dir / f"L{level}_{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {path}\n"
            "Run scripts/{dataset}/phase1_prepare/02_sample_seeds.py first."
        )
    with open(path, encoding="utf-8") as f:
        return {s["id"] for s in json.load(f)}


def filter_to_level(
    translated: list[dict],
    dataset: str,
    level: int,
    strategy: str = "greedy",
) -> list[dict]:
    """Return only the samples from `translated` that belong to Lk."""
    level_ids = load_level_ids(dataset, level, strategy)
    filtered = [s for s in translated if s["id"] in level_ids]
    if len(filtered) != len(level_ids):
        missing = len(level_ids) - len(filtered)
        print(
            f"  WARNING: {missing} IDs from L{level}_{strategy} not found in input file.",
            file=sys.stderr,
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a translated L5 file to any seed level Lk."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["spider", "bird"],
        help="Dataset name."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the fully-translated L5 JSON file."
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4, 5],
        help="Target level k (1–5). Mutually exclusive with --all-levels."
    )
    parser.add_argument(
        "--all-levels", action="store_true",
        help="Produce all levels L1..L5. Requires --output-dir."
    )
    parser.add_argument(
        "--output",
        help="Output file path (used with --level)."
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (used with --all-levels). Files named L{k}_{strategy}_vi.json."
    )
    parser.add_argument(
        "--strategy", default="greedy", choices=["greedy", "random_42", "random_123", "random_456"],
        help="Sampling strategy to use for level ID lists (default: greedy)."
    )
    args = parser.parse_args()

    # Validation
    if args.all_levels and args.level:
        parser.error("--level and --all-levels are mutually exclusive.")
    if not args.all_levels and not args.level:
        parser.error("Specify --level or --all-levels.")
    if args.all_levels and not args.output_dir:
        parser.error("--all-levels requires --output-dir.")
    if args.level and not args.output:
        parser.error("--level requires --output.")

    # Load translated L5
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        translated = json.load(f)
    print(f"Loaded {len(translated)} samples from {input_path.name}")

    strategy = args.strategy

    if args.level:
        # Single level
        filtered = filter_to_level(translated, args.dataset, args.level, strategy)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"L{args.level} ({strategy}): {len(filtered)} samples → {out}")

    else:
        # All levels
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 6):
            filtered = filter_to_level(translated, args.dataset, k, strategy)
            out = out_dir / f"L{k}_{strategy}_vi.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            print(f"L{k} ({strategy}): {len(filtered)} samples → {out.name}")


if __name__ == "__main__":
    main()
