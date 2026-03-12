# Phase 1: Data Extraction and Seed Sampling

## Overview

Phase 1 performs two operations:

1. **Extraction** — reads raw Spider and BIRD datasets into a unified internal format, assigns stable IDs, and computes SQL operator pattern signatures for each sample.
2. **Seed sampling** — selects a nested sequence of subsets at five seed ratios (5%, 10%, 15%, 20%, 25%) from the training split. These subsets serve as the human annotation targets in Phase 2 (gold seed construction).

The output of Phase 1 feeds directly into the Label Studio annotation workflow in Phase 2.

---

## 1. Extraction

### 1.1 Spider

| Input file | Description |
|---|---|
| `data/spider/train_spider.json` | 7,000 core training samples |
| `data/spider/train_others.json` | 1,659 samples adapted from GeoQuery, Scholar, etc. |
| `data/spider/dev.json` | 1,034 development samples |

Both training files are concatenated, re-indexed with stable IDs (`spider-train-00001`, ...), and written to `data/spider/extracted/train.json`. SQL operator patterns are computed and stored in the `sql_patterns` field of each sample.

**Output format:**

```json
{
  "id": "spider-train-00001",
  "db_id": "department_management",
  "question": "How many heads of the departments are older than 56?",
  "query": "SELECT count(*) FROM head WHERE age > 56",
  "sql_patterns": ["COUNT", "FROM", "SELECT", "WHERE"],
  "sql_class": "AGG_ONLY",
  "source": "spider"
}
```

**Spider train statistics after extraction:**

| Metric | Value |
|---|---|
| Total training samples | 8,659 |
| Unique databases | 146 |
| Unique SQL pattern combinations | 274 |

### 1.2 BIRD

| Input file | Description |
|---|---|
| `data/bird/train/train.json` | 9,428 training samples |
| `data/bird/dev/dev.json` | 1,534 development samples |

**Output format:**

```json
{
  "id": "bird-train-00001",
  "db_id": "movie_platform",
  "question": "Name movie titles released in year 1945. Sort by descending popularity.",
  "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
  "SQL": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1",
  "sql_patterns": ["FROM", "LIMIT", "ORDER BY", "SELECT", "WHERE"],
  "sql_class": "ORDER_LIMIT",
  "difficulty": null
}
```

> The `difficulty` field is `null` for training samples (only present in dev). The `evidence` field may be an empty string for ~7% of training samples.

**BIRD train statistics after extraction:**

| Metric | Value |
|---|---|
| Total training samples | 9,428 |
| Unique databases | 69 |
| Unique SQL pattern combinations | 720 |
| Non-empty evidence | 8,783 / 9,428 (93.2%) |

---

## 2. Dataset Partition

Before any annotation or translation begins, the full training set is partitioned into **three fixed regions**. This partition is determined once in Phase 1.

```
Train (100%)
├── Pool (45%) ─── fixed; split internally between manual and GPT per round
│   ├── L_k  (5k%, nested) ─── human-annotated; used as few-shot context for GPT
│   └── GPT_k = Pool \ L_k  ── GPT translates using L_k as few-shot context
└── Remaining (55%) ─────────── fine-tuned open-source model translates
```

### 2.1 The Five Rounds

At each round $k \in \{1,2,3,4,5\}$, the pool is split differently between manual and GPT:

| Round | Manual $L_k$ | GPT $\text{Pool} \setminus L_k$ | Fine-tune data | Remaining |
|---|---|---|---|---|
| 1 | 5% | 40% | **45%** | 55% |
| 2 | 10% | 35% | **45%** | 55% |
| 3 | 15% | 30% | **45%** | 55% |
| 4 | 20% | 25% | **45%** | 55% |
| 5 | 25% | 20% | **45%** | 55% |

The fine-tune data ($L_k \cup \text{GPT}_k$ = Pool = 45%) is **identical in size** across all rounds. This ensures that the fine-tuned open-source model always receives the same quantity of training data — quality differences in the Remaining translation are attributable solely to translation quality of the fine-tune data, not its size.

**Concrete sample counts:**

| Round | Spider manual | Spider GPT | BIRD manual | BIRD GPT |
|---|---|---|---|---|
| 1 | 433 | 3,464 | 472 | 3,771 |
| 2 | 866 | 3,031 | 943 | 3,300 |
| 3 | 1,299 | 2,598 | 1,415 | 2,828 |
| 4 | 1,732 | 2,165 | 1,886 | 2,357 |
| 5 | 2,165 | 1,732 | 2,357 | 1,886 |
| **Pool total** | **3,897** | | **4,243** | |
| **Remaining** | **4,762** | | **5,185** | |

### 2.2 Why This Design?

**Fixed fine-tune data size** eliminates confounds. If the open-source model received more training data at Round 5 than Round 1, a quality improvement could be explained by data quantity rather than translation quality. Fixing the pool size at 45% removes this confound.

**As $k$ increases:**
- $L_k$ grows → more high-quality manual translations in few-shot context → GPT quality improves
- $L_k$ grows → more high-quality manual translations directly in fine-tune data
- $\text{GPT}_k$ shrinks → less GPT-generated data in fine-tune data → less error propagation risk

The hypothesis is that these three effects together cause downstream performance to increase with $k$, and to plateau around $k=3$ or $k=4$.

**Zero-shot baseline ($k=0$):** GPT translates the entire pool (45%) with no few-shot examples; fine-tunes and evaluates identically. This is added as an additional reference point.

### 2.3 Nested Subset Design for $L_1 \subset L_2 \subset \cdots \subset L_5$

Within the pool ($L_5$, 25% of train), five nested annotation levels are defined:

$$L_1 \subset L_2 \subset L_3 \subset L_4 \subset L_5$$

Each level $L_k = L_{k-1} \cup \Delta_k$ where $\Delta_k$ is the new batch of samples added at round $k$. Human annotators complete batches sequentially — running the ablation at level $k$ only requires batches $1$ through $k$ to be annotated.

**GPT sets are also nested (reverse direction):**

$$\text{GPT}_5 \subset \text{GPT}_4 \subset \text{GPT}_3 \subset \text{GPT}_2 \subset \text{GPT}_1$$

because $\text{GPT}_k = \text{Pool} \setminus L_k$ and $L_k \subset L_{k+1}$. This means every round $k+1$ uses the *same* GPT samples as round $k$ (just fewer of them, with the smallest ones dropped). Combined quality differences between rounds are thus monotone and interpretable.

### 2.4 Sampling Algorithm

The pool (45%) and nested levels are selected using **greedy maximum coverage** over a single stratum dimension:

- `sql_class` — structural complexity class (10 classes; see taxonomy below)

Using a single dimension avoids the infeasibility problem of 3-way strata (db × hardness × sql_pattern), where low-ratio levels like L1 (5% ≈ 433 Spider samples) cannot cover all 274 pattern combinations. With only 10 sql_class values, even L1 achieves complete coverage.

**Tie-break:** when multiple candidates would cover the same number of new `sql_class` values, prefer the sample from the rarest `db_id` (fewest samples selected from that database so far). This incidentally promotes database diversity without making it the primary objective.

**Algorithm:**

```
function GREEDY_COVER(pool, budget, already_covered_strata):
    selected = []
    covered = already_covered_strata.copy()
    db_count = Counter()

    while len(selected) < budget and pool is not empty:
        uncovered = [s for s in pool if sql_class(s) not in covered]
        source = uncovered if uncovered else pool
        best = argmin_{s in source} db_count[db_id(s)]
        selected.append(best)
        covered.add(sql_class(best))
        db_count[db_id(best)] += 1
        pool.remove(best)

    return selected
```

**Full construction:**

```
# Step 1: Select Pool (45%) — covers as many sql_class strata as possible
Pool = GREEDY_COVER(train, budget=0.45*N, already_covered=∅)

# Step 2: Build nested seed levels within Pool
L1 = GREEDY_COVER(Pool, budget=0.05*N, already_covered=∅)
L2 = L1 ∪ GREEDY_COVER(Pool \ L1, budget=0.05*N, already_covered=sql_classes(L1))
L3 = L2 ∪ GREEDY_COVER(Pool \ L2, budget=0.05*N, already_covered=sql_classes(L2))
L4 = L3 ∪ GREEDY_COVER(Pool \ L3, budget=0.05*N, already_covered=sql_classes(L3))
L5 = L4 ∪ GREEDY_COVER(Pool \ L4, budget=0.05*N, already_covered=sql_classes(L4))
# L5 = full seed pool (25%)

# Step 3: GPT sets are derived automatically
GPT_k = Pool \ L_k   (for k = 0..5, where L_0 = ∅)

# Step 4: Remaining is fixed
Remaining = train \ Pool
```

### 2.5 SQL Class Taxonomy

Each sample is assigned a `sql_class` label by `scripts/utils/sql_validator.classify_sql()` using a priority-based rule:

| Class | Description | Priority |
|---|---|---|
| `SET_OP` | UNION / INTERSECT / EXCEPT | 1 (highest) |
| `NESTED` | One or more subqueries (SELECT inside SELECT) | 2 |
| `GROUP_HAVING` | GROUP BY + HAVING | 3 |
| `GROUP_BY` | GROUP BY without HAVING | 4 |
| `JOIN_ORDER` | JOIN + ORDER BY or LIMIT | 5 |
| `JOIN` | JOIN, no ORDER BY / LIMIT | 6 |
| `ORDER_LIMIT` | ORDER BY / LIMIT, no JOIN | 7 |
| `AGG_ONLY` | Aggregate functions (COUNT, SUM, …), no JOIN | 8 |
| `SELECT_WHERE` | SELECT + WHERE, no join/aggregate | 9 |
| `SIMPLE` | Bare SELECT, no WHERE clause | 10 (lowest) |

**Empirical distribution (train sets):**

| Class | Spider | BIRD |
|---|---|---|
| `JOIN` | 2,145 (24.8%) | 5,247 (55.7%) |
| `GROUP_BY` | 1,469 (17.0%) | 702 (7.4%) |
| `SELECT_WHERE` | 1,071 (12.4%) | 689 (7.3%) |
| `AGG_ONLY` | 970 (11.2%) | 866 (9.2%) |
| `NESTED` | 824 (9.5%) | 714 (7.6%) |
| `ORDER_LIMIT` | 661 (7.6%) | 256 (2.7%) |
| `SET_OP` | 526 (6.1%) | 30 (0.3%) |
| `GROUP_HAVING` | 407 (4.7%) | 86 (0.9%) |
| `SIMPLE` | 342 (3.9%) | 6 (0.1%) |
| `JOIN_ORDER` | 244 (2.8%) | 832 (8.8%) |

### 2.6 Coverage Capacity Analysis

Because there are only 10 `sql_class` values, even L1 (≈5% of train) achieves **complete class coverage** on both datasets. This was verified after running the sampler:

#### Spider

| Partition | Samples | DBs covered (146) | sql_class covered (10) |
|---|---|---|---|
| Pool (45%) | 3,897 | 146 (full) | 10 (full) |
| L1 (5%) | 433 | 146 (full) | 10 (full) |
| L2 (10%) | 866 | 146 (full) | 10 (full) |
| L5 (25%) | 2,165 | 146 (full) | 10 (full) |
| Remaining (55%) | 4,762 | full | full |

#### BIRD

| Partition | Samples | DBs covered (69) | sql_class covered (10) |
|---|---|---|---|
| Pool (45%) | 4,243 | 69 (full) | 10 (full) |
| L1 (5%) | 472 | 69 (full) | 10 (full) |
| L2 (10%) | 943 | 69 (full) | 10 (full) |
| L5 (25%) | 2,357 | 69 (full) | 10 (full) |
| Remaining (55%) | 5,185 | full | full |

Full `sql_class` coverage at L1 means that the few-shot context for GPT always includes at least one example of each structural complexity class, even in the lowest-seed round.

### 2.7 Hardness/Difficulty Field

| Dataset | Train label | Source |
|---|---|---|
| Spider | Not in raw JSON | Inferred at extraction using same rule-based classifier as official Spider eval script |
| BIRD | `difficulty` absent in `train.json` | Approximated by the same structural heuristic, consistent with dev set labels |

### 2.8 Sampling Strategy Comparison

To test whether *how* samples are selected matters (not just *how many*), a **random baseline** is also evaluated:

| Strategy | Description |
|---|---|
| `greedy` | Greedy maximum coverage — primary strategy |
| `random` | Uniform random sampling without replacement — baseline |

For random, `n_runs=3` independent draws are run at each ratio to estimate variance.

---

## 3. Output Files

```
data/spider/extracted/
├── train.json              # 8,659 samples with stable IDs and sql_patterns
├── dev.json                # 1,034 samples
└── seeds/
    ├── partition.json      # Single source of truth: each sample → partition + seed_level
    ├── pool_greedy.json    # 3,897 samples — full pool (45%), greedy
    ├── L1_greedy.json      # 433 samples  — Level 1 seed (greedy)
    ├── L2_greedy.json      # 866 samples  — Level 2 seed (L1 ⊂ L2)
    ├── L3_greedy.json      # 1,299 samples
    ├── L4_greedy.json      # 1,732 samples
    ├── L5_greedy.json      # 2,165 samples — full seed (= L5 ⊂ Pool)
    ├── remaining.json      # 4,762 samples — for open-source model translation
    ├── pool_random_{seed}.json    # random baseline variants
    └── L1_random_{seed}.json
    └── ...

data/bird/extracted/
├── train.json
├── dev.json
└── seeds/
    ├── partition.json
    ├── pool_greedy.json    # 4,243 samples
    ├── L1_greedy.json      # 472 samples
    ├── ...
    ├── L5_greedy.json      # 2,357 samples
    └── remaining.json      # 5,185 samples
```

**`partition.json` format** — single source of truth for all downstream phases:

```json
{
  "spider-train-00042": {
    "partition": "seed",
    "seed_level": 1
  },
  "spider-train-00107": {
    "partition": "pool"
  },
  "spider-train-00215": {
    "partition": "remaining"
  }
}
```

`partition` values:
- `"seed"` — in L5 (human-annotated); always also part of Pool
- `"pool"` — in Pool but not in L5 (GPT-translated at every round)
- `"remaining"` — 55%; translated by fine-tuned open-source model

`seed_level`: `1`–`5` for seed samples (first round this sample is annotated), absent otherwise.

At Phase 3 round $k$, the GPT input is trivially derived as:
```python
GPT_k = [s for s in pool if partition[s.id] != "seed" or s.seed_level > k]
```

---

## 4. Scripts

| Script | Description |
|---|---|
| `scripts/spider/phase1_prepare/01_extract.py` | Extract and normalize Spider data |
| `scripts/spider/phase1_prepare/02_sample_seeds.py` | Generate nested seed subsets for Spider |
| `scripts/bird/phase1_prepare/01_extract.py` | Extract and normalize BIRD data |
| `scripts/bird/phase1_prepare/02_sample_seeds.py` | Generate nested seed subsets for BIRD |

**Run Phase 1:**

```bash
# Spider
python scripts/spider/phase1_prepare/01_extract.py
python scripts/spider/phase1_prepare/02_sample_seeds.py

# BIRD
python scripts/bird/phase1_prepare/01_extract.py
python scripts/bird/phase1_prepare/02_sample_seeds.py
```

---

## 5. Design Rationale

**Why fix the pool size at 45% across all rounds?**

The fine-tuned open-source model always receives the same 45% of training data regardless of round. This ensures that quality differences in the Remaining translation (and downstream NL2SQL performance) are attributable to *translation quality* of the fine-tune data, not its *size*. If round 5 used more data than round 1, an improvement could be explained by having a larger fine-tuning set.

**Why do both $L_k$ and $\text{GPT}_k$ change with $k$?**

Unlike a design that fixes the GPT samples and only modifies few-shot context, here as more samples become manually translated ($L_k$ grows), those samples directly enter the fine-tuning data at their highest possible quality — replacing the corresponding GPT-translated versions. This makes the design more realistic: in practice, more annotation budget means not only better few-shot context but also higher-quality training data. The combined effect is what the research pipeline measures.

**Why nested subsets for $L_1 \subset \cdots \subset L_5$ (and $\text{GPT}_5 \subset \cdots \subset \text{GPT}_1$)?**

If levels were independently sampled, a quality difference between rounds could be due to sampling luck. With the nested design, $L_{k+1} = L_k \cup \Delta_{k+1}$ ensures any improvement is attributable purely to the additional annotations $\Delta_{k+1}$, not to which samples happened to be selected.

**Why greedy maximum coverage rather than stratified proportional sampling?**

In Spider and BIRD, easy examples and common SQL patterns are heavily over-represented. Proportional sampling would waste annotation budget on redundant easy examples while under-sampling rare patterns — the opposite of what few-shot prompting needs. Greedy coverage actively seeks diversity across DB, difficulty, and SQL pattern dimensions within a fixed budget.

**Why compare random vs. greedy sampling strategies?**

If `random ≈ greedy` across all levels, any $N$ samples are approximately equivalent as few-shot context, and the annotation workflow can use simple random sampling. If `greedy > random`, it motivates investing in the coverage-based selection tool. This is a secondary finding with practical implications for future low-resource translation pipelines.

**Why use a multilingual NL2SQL model for Phase 5 evaluation?**

Fine-tuning the same multilingual model on EN and VI data with identical hyperparameters and comparing Execution Accuracy (and Exact Match) isolates translation quality as the independent variable. A multilingual base model with cross-lingual representations minimizes the risk that VI performance drops merely because the model was pre-trained predominantly on English — making the EN vs. VI comparison attributable to dataset quality rather than model architecture.
