"""
Phase 1 — BIRD Seed Sampling (02_sample_seeds.py)

Reads the extracted BIRD training set and produces:
  - A fixed Pool (45%) using greedy maximum-coverage sampling
  - Nested annotation levels L1 ⊂ L2 ⊂ L3 ⊂ L4 ⊂ L5 (5%→25% of train)
    built within the pool, also greedily
  - A random baseline (n_runs=3) for comparison

BIRD train.json does not contain a difficulty field (only dev does).
An approximate difficulty label is inferred from SQL structure heuristics —
the same approach used for dev annotation, so the two sets are comparable.

At each ablation round k:
  - L_k  (5k%)         — human-annotated, as GPT few-shot context
  - Pool \ L_k (45%-5k%) — GPT-translated, using L_k as few-shot
  - Remaining (55%)    — translated by the fine-tuned open-source model
  Fine-tune data = Pool (45%) — constant across all rounds.

Input:
    data/bird/extracted/train.json

Output (data/bird/extracted/seeds/):
    partition.json          — source of truth: id → {partition, seed_level}
    pool_greedy.json        — 3,772 samples (Pool, greedy)
    L{1..5}_greedy.json     — nested seed levels (greedy)
    remaining.json          — 5,656 samples (55%, open-source model)
    pool_random_{seed}.json — random baseline variants
    L{1..5}_random_{seed}.json
    sampling_stats.json     — coverage statistics
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sampler import (
    build_nested_levels,
    get_stratum_key,
    greedy_cover,
    infer_difficulty_bird,
    random_cover,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POOL_RATIO = 0.45
SEED_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25]  # L1 … L5
RANDOM_SEEDS = [42, 123, 456]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "bird" / "extracted"
SEEDS_DIR = EXTRACTED_DIR / "seeds"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train() -> list[dict]:
    path = EXTRACTED_DIR / "train.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    n = len(data) if isinstance(data, (list, dict)) else "?"
    print(f"  Wrote {n} → {path.relative_to(PROJECT_ROOT)}")


def coverage_stats(samples: list[dict], label: str) -> dict:
    dbs = {s["db_id"] for s in samples}
    diff = Counter(s.get("hardness", "?") for s in samples)
    strata = {get_stratum_key(s) for s in samples}
    evidence_nonempty = sum(1 for s in samples if s.get("evidence", ""))
    return {
        "label": label,
        "n": len(samples),
        "unique_dbs": len(dbs),
        "difficulty_distribution": dict(diff),
        "unique_strata": len(strata),
        "evidence_nonempty": evidence_nonempty,
        "evidence_ratio": round(evidence_nonempty / len(samples), 4) if samples else 0,
    }


def build_partition(
    train: list[dict],
    pool_ids: set[str],
    seed_levels: list[list[dict]],
) -> dict:
    id_to_level: dict[str, int] = {}
    seen: set[str] = set()
    for k, level_samples in enumerate(seed_levels, start=1):
        for s in level_samples:
            if s["id"] not in seen:
                id_to_level[s["id"]] = k
                seen.add(s["id"])

    partition: dict[str, dict] = {}
    for s in train:
        sid = s["id"]
        if sid in id_to_level:
            partition[sid] = {"partition": "seed", "seed_level": id_to_level[sid]}
        elif sid in pool_ids:
            partition[sid] = {"partition": "pool"}
        else:
            partition[sid] = {"partition": "remaining"}
    return partition


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("BIRD Phase 1 — Seed Sampling")
    print("=" * 60)

    # --- Load data ---
    train = load_train()
    N = len(train)
    print(f"\nLoaded {N} BIRD training samples.")

    # --- Add inferred difficulty (used as 'hardness' in stratum key) ---
    for s in train:
        s["hardness"] = infer_difficulty_bird(s)

    diff_dist = Counter(s["hardness"] for s in train)
    print(f"Difficulty distribution (inferred): {dict(diff_dist)}")

    # --- Budgets ---
    pool_budget = math.ceil(N * POOL_RATIO)
    seed_budgets = [math.ceil(N * r) for r in SEED_RATIOS]

    print(f"\nBudgets:")
    print(f"  Pool (45%):      {pool_budget:,}")
    for i, (r, b) in enumerate(zip(SEED_RATIOS, seed_budgets), 1):
        print(f"  L{i} ({r*100:.0f}%):       {b:,}")
    print(f"  Remaining (55%): {N - pool_budget:,}")

    # --- Greedy: select pool ---
    print("\nSelecting Pool (greedy)...")
    pool_samples, pool_strata = greedy_cover(
        pool=train,
        budget=pool_budget,
    )
    pool_ids = {s["id"] for s in pool_samples}
    remaining_samples = [s for s in train if s["id"] not in pool_ids]

    print(f"  Pool: {len(pool_samples):,} samples, {len(pool_strata):,} strata covered")

    # --- Greedy: nested seed levels within pool ---
    print("Building nested seed levels (greedy)...")
    seed_levels_greedy = build_nested_levels(
        pool=pool_samples,
        level_sizes=seed_budgets,
    )

    for i, level in enumerate(seed_levels_greedy, 1):
        ldb = len({s["db_id"] for s in level})
        lstrata = len({get_stratum_key(s) for s in level})
        print(f"  L{i}: {len(level):,} samples, {ldb} DBs, {lstrata} strata")

    # --- Random baselines ---
    print(f"\nBuilding random baselines ({len(RANDOM_SEEDS)} runs)...")
    random_pool_runs: list[list[dict]] = []
    random_seed_level_runs: list[list[list[dict]]] = []

    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        shuffled_train = list(train)
        rng.shuffle(shuffled_train)
        rand_pool = shuffled_train[:pool_budget]
        random_pool_runs.append(rand_pool)

        rand_levels = random_cover(rand_pool, seed_budgets, seed=seed)
        random_seed_level_runs.append(rand_levels)
        print(f"  seed={seed}: pool={len(rand_pool):,} samples")

    # --- Build partition.json ---
    partition = build_partition(train, pool_ids, seed_levels_greedy)

    seed_count = sum(1 for v in partition.values() if v["partition"] == "seed")
    pool_only_count = sum(1 for v in partition.values() if v["partition"] == "pool")
    remaining_count = sum(1 for v in partition.values() if v["partition"] == "remaining")
    print(f"\nPartition summary:")
    print(f"  seed (L5):  {seed_count:,}")
    print(f"  pool only:  {pool_only_count:,}")
    print(f"  remaining:  {remaining_count:,}")
    print(f"  total:      {seed_count + pool_only_count + remaining_count:,}")

    # --- Coverage statistics ---
    all_strata = {get_stratum_key(s) for s in train}
    stats = {
        "dataset": "bird",
        "train_n": N,
        "all_strata": len(all_strata),
        "splits": [
            coverage_stats(train, "train (full)"),
            coverage_stats(pool_samples, "pool_greedy (45%)"),
            *[coverage_stats(seed_levels_greedy[i], f"L{i+1}_greedy") for i in range(5)],
            coverage_stats(remaining_samples, "remaining (55%)"),
        ],
    }

    # --- Save ---
    print("\nSaving outputs...")
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)

    save_json(partition, SEEDS_DIR / "partition.json")
    save_json(pool_samples, SEEDS_DIR / "pool_greedy.json")
    for i, level in enumerate(seed_levels_greedy, 1):
        save_json(level, SEEDS_DIR / f"L{i}_greedy.json")
    save_json(remaining_samples, SEEDS_DIR / "remaining.json")
    save_json(stats, SEEDS_DIR / "sampling_stats.json")

    for seed, rand_pool, rand_levels in zip(RANDOM_SEEDS, random_pool_runs, random_seed_level_runs):
        save_json(rand_pool, SEEDS_DIR / f"pool_random_{seed}.json")
        for i, level in enumerate(rand_levels, 1):
            save_json(level, SEEDS_DIR / f"L{i}_random_{seed}.json")

    print(f"\nDone. All outputs in {SEEDS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
