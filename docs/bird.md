# BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL

## Overview

BIRD (BIg bench for laRge-scale Database grounded text-to-SQL) is a large-scale Text-to-SQL benchmark introduced by Li et al. at NeurIPS 2023. Unlike Spider, which focuses on schema-level generalization, BIRD emphasizes **database value comprehension** — the ability to reason over actual database contents in order to generate correct SQL. This reflects a more realistic setting in which domain-specific knowledge encoded in cell values, column semantics, and complex expressions is necessary to resolve ambiguity in natural language questions.

**Paper:** *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*  
**Authors:** Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C. C. Chang, Fei Huang, Reynold Cheng, Yongbin Li  
**Venue:** NeurIPS 2023  
**arXiv:** https://arxiv.org/abs/2305.03111  
**Project page / Leaderboard:** https://bird-bench.github.io/

---

## Dataset Statistics

| Split | Samples | Databases | Notes |
|---|---|---|---|
| `train` | 9,428 | 69 | Training set; no `question_id` or `difficulty` fields |
| `dev` | 1,534 | 11 | Validation set; includes `question_id` and `difficulty` |
| `test` | ~1,789 | 15 | **Not publicly released**; submitted to leaderboard |
| **Total (public)** | **10,962** | **80** | |

**Scale:** 95 databases total (including test), total database size ~33.4 GB, spanning 37 professional domains.

**Difficulty distribution (dev set):**

| Level | Count | Percentage |
|---|---|---|
| Simple | 925 | 60.3% |
| Moderate | 464 | 30.2% |
| Challenging | 145 | 9.5% |

**Schema statistics:**

| Metric | Train DBs (69) | Dev DBs (11) |
|---|---|---|
| Average tables per database | 7.6 | — |
| Average columns per database | 52.3 | — |
| Total columns | 3,608 | 809 |

**Evidence coverage:**

| Split | Samples with non-empty `evidence` | Percentage |
|---|---|---|
| Train | 8,783 / 9,428 | 93.1% |
| Dev | 1,386 / 1,534 | 90.4% |

---

## Data Format

### Training set (`train/train.json`)

Each sample is a JSON object with four fields:

```json
{
  "db_id": "movie_platform",
  "question": "Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.",
  "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
  "SQL": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"
}
```

### Development set (`dev/dev.json`)

The dev set has two additional fields:

```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
  "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
  "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
  "difficulty": "simple"
}
```

| Field | Present in | Type | Description |
|---|---|---|---|
| `question_id` | dev only | `int` | Unique identifier for the sample |
| `db_id` | train + dev | `string` | Identifier of the target database |
| `question` | train + dev | `string` | Natural language question in English |
| `evidence` | train + dev | `string` | External domain knowledge to assist SQL generation; may be empty |
| `SQL` | train + dev | `string` | Ground-truth SQL query |
| `difficulty` | dev only | `string` | Difficulty label: `simple`, `moderate`, or `challenging` |

> **Important — `evidence` field:** The `evidence` field provides external knowledge that bridges the gap between the natural language question and the database schema or values. It commonly contains arithmetic definitions (e.g., "rate = A / B"), domain-specific terminology mappings, or filtering conditions expressed in natural language. In downstream Text-to-SQL models, `evidence` is typically concatenated with the question as additional context. For translation purposes, `evidence` must be translated together with `question` in the same API call to ensure terminological consistency.

---

## Schema Format (`train_tables.json`, `dev_tables.json`)

The schema format mirrors Spider's `tables.json`:

```json
{
  "db_id": "debit_card_specializing",
  "table_names_original": ["customers", "gasstations", "products", "transactions_1k", "yearmonth"],
  "table_names": ["customers", "gas stations", "products", "transactions", "year and month"],
  "column_names_original": [[-1, "*"], [0, "CustomerID"], [0, "Segment"], ...],
  "column_names": [[-1, "*"], [0, "customer id"], [0, "segment"], ...],
  "column_types": ["text", "number", "text", ...],
  "primary_keys": [...],
  "foreign_keys": [[...], ...]
}
```

| Field | Description |
|---|---|
| `table_names_original` | Exact table names as stored in the SQLite database |
| `table_names` | Human-readable normalized table names |
| `column_names_original` | `[table_index, column_name]` pairs using original names |
| `column_names` | `[table_index, column_name]` pairs using normalized names |
| `column_types` | Data type per column (`text`, `number`, `time`, `boolean`, `others`) |
| `primary_keys` | Column indices that are primary keys |
| `foreign_keys` | Pairs of column indices representing foreign key relationships |

---

## SQL Difficulty Levels

Unlike Spider (which computes hardness programmatically from SQL structure), BIRD assigns difficulty labels manually during annotation, based on the combination of SQL complexity and the degree of domain knowledge required to construct the query.

| Level | Description |
|---|---|
| **Simple** | Direct mapping from question to SQL; minimal domain knowledge required; basic SELECT/WHERE |
| **Moderate** | Requires moderate domain knowledge or multi-table JOINs; some aggregation or filtering |
| **Challenging** | Requires deep domain reasoning, complex arithmetic, nested subqueries, or multi-step computation via the `evidence` field |

---

## Evaluation Metrics

BIRD introduces two metrics:

### 1. Execution Accuracy (EX)

The percentage of predicted SQL queries that produce a result set identical to the gold SQL when executed against the database. This is the primary metric for correctness.

$$\text{EX} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{Exec}(P_i, D_i) = \text{Exec}(G_i, D_i)]$$

where $P_i$ is the predicted SQL, $G_i$ is the gold SQL, and $D_i$ is the database.

### 2. Valid Efficiency Score (VES)

A secondary metric that evaluates not only whether the SQL is correct but also how efficiently it runs. VES penalizes queries that are unnecessarily slow compared to the gold SQL:

$$\text{VES} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{Exec}(P_i) = \text{Exec}(G_i)] \cdot \sqrt{\frac{R(G_i)}{R(P_i)}}$$

where $R(\cdot)$ denotes the execution time ratio. VES is particularly relevant for production systems operating on large databases.

> For this research project, **EX is the primary metric**. VES is reported as supplementary information.

---

## Key Design Properties

1. **Database value comprehension:** Many questions cannot be answered without understanding actual cell values. For example, knowing that a `segment` column stores values like `LAM`, `KAM`, `SME` requires domain knowledge unavailable from the schema alone.
2. **External knowledge via `evidence`:** The `evidence` field encodes domain-specific formulas, terminology definitions, and implicit constraints that connect natural language phrasing to SQL constructs.
3. **Realistic database scale:** Databases are large (up to several GB), with complex, real-world schemas averaging 7.6 tables and 52.3 columns per database — significantly more complex than Spider (5.3 tables, 28.1 columns).
4. **Dirty and ambiguous data:** Databases may contain inconsistent values, null entries, and domain-specific encodings that require careful reasoning.
5. **No test set leakage:** The test set is not publicly released; evaluation is conducted by submitting predictions to the official leaderboard at https://bird-bench.github.io/.

---

## Comparison with Spider

| Property | Spider | BIRD |
|---|---|---|
| Year | 2018 | 2023 |
| Venue | EMNLP | NeurIPS |
| Train samples | 8,659 | 9,428 |
| Dev samples | 1,034 | 1,534 |
| Total databases (public) | 166 | 80 |
| Avg tables per DB | 5.3 | 7.6 |
| Avg columns per DB | 28.1 | 52.3 |
| Domains | 138 | 37 |
| DB size | Small (MB range) | Large (~33.4 GB total) |
| External knowledge field | ✗ | ✓ (`evidence`) |
| Difficulty annotation | Computed (structural) | Manual (semantic) |
| Difficulty levels | easy / medium / hard / extra hard | simple / moderate / challenging |
| Primary metric | Exact Match + EX | EX |
| Secondary metric | — | VES |
| Public test set | ✓ (no gold SQL) | ✗ (leaderboard only) |

---

## File Structure (in this repository)

```
data/bird/
├── train/
│   ├── train.json              # 9,428 training samples
│   ├── train_tables.json       # Schema definitions for 69 training databases
│   ├── train_gold.sql          # Gold SQL queries (one per line)
│   └── train_databases.zip     # SQLite databases for training
└── dev/
    ├── dev.json                # 1,534 development samples (with difficulty + question_id)
    ├── dev_tables.json         # Schema definitions for 11 dev databases
    ├── dev_tied_append.json    # Additional dev samples for tie-breaking evaluation
    ├── dev.sql                 # Gold SQL queries for dev set
    └── dev_databases.zip       # SQLite databases for development
```

---

## Citation

```bibtex
@inproceedings{Li2023BIRD,
  title     = {Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author    = {Jinyang Li and Binyuan Hui and Ge Qu and Jiaxi Yang and Binhua Li and Bowen Li and Bailin Wang and Bowen Qin and Rongyu Cao and Ruiying Geng and Nan Huo and Xuanhe Zhou and Chenhao Ma and Guoliang Li and Kevin C. C. Chang and Fei Huang and Reynold Cheng and Yongbin Li},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```
