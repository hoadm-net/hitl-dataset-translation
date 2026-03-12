"""
Greedy maximum-coverage sampler for Phase 1 seed selection.

Used by Phase 1 (02_sample_seeds.py) for both Spider and BIRD.

Each sample is associated with a single stratum key:  sql_class
(see sql_validator.classify_sql for the 10-class taxonomy).

The greedy algorithm iteratively picks the sample that adds the most new
sql_class strata to the covered set.  Tie-breaking uses db_id frequency
(prefer rarer databases), which incidentally promotes DB diversity without
making it the primary sampling objective.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Stratum helper — single dimension
# ---------------------------------------------------------------------------

def get_stratum_key(sample: dict) -> str:
    """Return the sql_class string as the stratum key for a sample."""
    return sample["sql_class"]


# ---------------------------------------------------------------------------
# Hardness inference (kept for optional secondary annotation)
# ---------------------------------------------------------------------------

def infer_hardness_spider(sample: dict) -> str:
    """
    Rule-based Spider hardness: easy / medium / hard / extra.

    Approximates the official Spider eval-script classification using
    sql_patterns (pre-computed) and a raw SQL string scan.
    """
    sql = sample.get("query", "").upper()
    patterns = set(sample.get("sql_patterns", []))

    # Subquery count: every SELECT beyond the first is a nested query
    subquery_count = sql.count("SELECT") - 1

    # JOIN count (LEFT JOIN / INNER JOIN also count as a JOIN)
    join_count = sql.count(" JOIN ")

    has_set_ops = bool(patterns & {"INTERSECT", "EXCEPT", "UNION"})
    has_agg = bool(patterns & {"COUNT", "SUM", "AVG", "MAX", "MIN"})
    has_complex = bool(patterns & {"HAVING", "GROUP BY", "ORDER BY"})
    has_where = "WHERE" in patterns

    # Extra: set operations or ≥3 nested subqueries
    if has_set_ops or subquery_count >= 3:
        return "extra"

    # Hard: 2+ nested subqueries, or subquery + complex join
    if subquery_count >= 2 or (subquery_count >= 1 and join_count >= 2):
        return "hard"

    # Medium: any join, aggregation, GROUP/ORDER/HAVING, or single subquery
    if subquery_count >= 1 or has_agg or has_complex or join_count >= 1:
        return "medium"

    return "easy"


def infer_difficulty_bird(sample: dict) -> str:
    """
    Approximate BIRD difficulty: simple / moderate / challenging.

    BIRD train.json does not contain a difficulty field; this heuristic
    mimics the structural complexity used for the dev annotations.
    Evidence complexity is not considered (text only, no impact on SQL).
    """
    sql = sample.get("SQL", "").upper()
    patterns = set(sample.get("sql_patterns", []))

    subquery_count = sql.count("SELECT") - 1
    join_count = sql.count(" JOIN ")

    has_set_ops = bool(patterns & {"INTERSECT", "EXCEPT", "UNION"})
    has_agg = bool(patterns & {"COUNT", "SUM", "AVG", "MAX", "MIN"})
    has_complex = bool(patterns & {"HAVING", "GROUP BY", "ORDER BY"})

    # Challenging: set operations or 2+ nested subqueries
    if has_set_ops or subquery_count >= 2:
        return "challenging"

    # Moderate: join, aggregation, GROUP/ORDER/HAVING, single subquery
    if subquery_count >= 1 or has_agg or has_complex or join_count >= 1:
        return "moderate"

    return "simple"


# ---------------------------------------------------------------------------
# Greedy maximum-coverage sampler
# ---------------------------------------------------------------------------

def greedy_cover(
    pool: list[dict],
    budget: int,
    covered_strata: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """
    Select up to *budget* samples from *pool* using greedy maximum coverage.

    Primary objective: maximize coverage of sql_class strata.
    Tie-break: prefer samples from the rarest db_id in the current selection
    (promotes DB diversity without making it the primary objective).

    Args:
        pool: Candidate samples.  Must have 'sql_class' and 'db_id'.
        budget: Maximum number of samples to return.
        covered_strata: sql_class values already covered (read-only; copy made).

    Returns:
        (selected, updated_covered_strata) where updated_covered_strata
        adds the strata of all selected samples.
    """
    if covered_strata is None:
        covered_strata = set()
    covered: set[str] = set(covered_strata)  # working copy

    remaining = list(pool)
    selected: list[dict] = []
    db_count: Counter = Counter()  # db_ids chosen so far → tiebreak metric

    # Group into uncovered-class candidates and already-covered candidates
    # (recalculated each step would be O(n²); pre-group for O(n log n) approx)
    # Since budget ≤ len(pool) ≤ ~10k this simple approach is fast enough.
    while remaining and len(selected) < budget:
        # Prefer samples that introduce a new sql_class
        uncovered = [s for s in remaining if s["sql_class"] not in covered]
        source = uncovered if uncovered else remaining

        # From the chosen group, pick the one from the rarest db
        best = min(source, key=lambda s: (db_count[s["db_id"]], s["id"]))

        selected.append(best)
        covered.add(best["sql_class"])
        db_count[best["db_id"]] += 1
        remaining.remove(best)

    return selected, covered


# ---------------------------------------------------------------------------
# Incremental nested-level builder
# ---------------------------------------------------------------------------

def build_nested_levels(
    pool: list[dict],
    level_sizes: list[int],
) -> list[list[dict]]:
    """
    Build nested levels L1 ⊂ L2 ⊂ … ⊂ L_n within *pool*.

    level_sizes[k] is the TOTAL cumulative size of L_{k+1}, not the increment.

    Args:
        pool: The full pool from which all levels are drawn (not mutated).
        level_sizes: Non-decreasing list of cumulative sizes, e.g. [433, 866, …].

    Returns:
        List of *n* sample lists where result[k] = L_{k+1} (cumulative).
        Each is a superset of the previous.
    """
    assert level_sizes == sorted(level_sizes), "level_sizes must be non-decreasing"
    assert level_sizes[-1] <= len(pool), (
        f"Largest level ({level_sizes[-1]}) exceeds pool size ({len(pool)})"
    )

    levels: list[list[dict]] = []
    covered: set[str] = set()
    pool_remaining = list(pool)

    for k, target_size in enumerate(level_sizes):
        prev_size = len(levels[k - 1]) if k > 0 else 0
        increment = target_size - prev_size

        new_samples, covered = greedy_cover(
            pool=pool_remaining,
            budget=increment,
            covered_strata=covered,
        )

        # Remove selected samples so later increments don't re-pick them
        selected_ids = {s["id"] for s in new_samples}
        pool_remaining = [s for s in pool_remaining if s["id"] not in selected_ids]

        cumulative = (levels[k - 1] if k > 0 else []) + new_samples
        levels.append(cumulative)

    return levels


# ---------------------------------------------------------------------------
# Random baseline sampler
# ---------------------------------------------------------------------------

def random_cover(
    pool: list[dict],
    level_sizes: list[int],
    seed: int,
) -> list[list[dict]]:
    """
    Build nested levels using uniform random sampling (no coverage bias).

    Args:
        pool: Full pool.
        level_sizes: Cumulative level sizes (same format as build_nested_levels).
        seed: Random seed for reproducibility.

    Returns:
        Same structure as build_nested_levels.
    """
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    levels = []
    for target_size in level_sizes:
        levels.append(list(shuffled[:target_size]))
    return levels
