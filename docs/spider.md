# Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL

## Overview

Spider is a large-scale, complex, and cross-domain semantic parsing and Text-to-SQL benchmark introduced by Yu et al. at EMNLP 2018. The dataset was annotated by 11 college students and is designed to evaluate the generalization capability of models across new SQL queries and unseen database schemas simultaneously — a property that distinguishes it from prior single-domain benchmarks such as GeoQuery and ATIS.

**Paper:** *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*  
**Authors:** Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev  
**Venue:** EMNLP 2018  
**arXiv:** https://arxiv.org/abs/1809.08887  
**Project page:** https://yale-lily.github.io/spider

---

## Dataset Statistics

| Split | Samples | Databases | Notes |
|---|---|---|---|
| `train_spider` | 7,000 | 140 | Core training set, originally annotated |
| `train_others` | 1,659 | 6 | Adapted from Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp |
| **Train (total)** | **8,659** | **146** | Combined for training |
| `dev` | 1,034 | 20 | Validation set |
| `test` | 2,147 | — | Split held by organizers; databases not distributed |
| **Total** | **166** databases | — | Across 138 domains |

**Schema statistics (from `tables.json`, 166 databases):**

| Metric | Value |
|---|---|
| Total databases | 166 |
| Average tables per database | 5.3 |
| Average columns per database | 28.1 |
| Total columns across all databases | 4,669 |

---

## Data Format

Each sample in `train_spider.json` and `dev.json` is a JSON object with the following fields:

```json
{
  "db_id": "department_management",
  "question": "How many heads of the departments are older than 56?",
  "question_toks": ["How", "many", "heads", "of", "the", "departments", "are", "older", "than", "56", "?"],
  "query": "SELECT count(*) FROM head WHERE age > 56",
  "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "head", "WHERE", "age", ">", "56"],
  "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "head", "where", "age", ">", "value"],
  "sql": { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `db_id` | `string` | Identifier of the target database |
| `question` | `string` | Natural language question in English |
| `question_toks` | `list[str]` | Tokenized question |
| `query` | `string` | Ground-truth SQL query |
| `query_toks` | `list[str]` | Tokenized SQL query |
| `query_toks_no_value` | `list[str]` | Tokenized SQL with literal values replaced by `value` |
| `sql` | `object` | Structured parsed representation of the SQL query |

> **Note:** The `hardness` label (easy / medium / hard / extra hard) is not stored in the data files. It is computed at evaluation time by the official Spider evaluation script (`evaluation.py`) based on the structural complexity of the SQL query (number of components, nesting depth, etc.).

---

## Schema Format (`tables.json`)

Database schemas are stored in `tables.json`. Each entry describes one database:

```json
{
  "db_id": "concert_singer",
  "table_names_original": ["stadium", "singer", "concert", "singer_in_concert"],
  "table_names": ["stadium", "singer", "concert", "singer in concert"],
  "column_names_original": [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...],
  "column_names": [[-1, "*"], [0, "stadium id"], [0, "location"], ...],
  "column_types": ["text", "number", "text", ...],
  "primary_keys": [1, 6, ...],
  "foreign_keys": [[12, 1], [14, 6], ...]
}
```

| Field | Description |
|---|---|
| `table_names_original` | Original table names as in the SQLite database |
| `table_names` | Normalized/human-readable table names |
| `column_names_original` | `[table_index, column_name]` pairs using original names |
| `column_names` | `[table_index, column_name]` pairs using normalized names |
| `column_types` | Data type for each column (`text`, `number`, `time`, `boolean`, `others`) |
| `primary_keys` | Column indices (from `column_names`) that are primary keys |
| `foreign_keys` | Pairs of column indices representing foreign key relationships |

---

## SQL Hardness Categories

The official Spider evaluation script classifies queries into four hardness levels based on structural complexity:

| Level | Description |
|---|---|
| **Easy** | Single SELECT clause, no JOIN, no subquery, no GROUP BY |
| **Medium** | Simple JOINs or WHERE conditions, basic aggregation |
| **Hard** | Multiple JOINs, GROUP BY/HAVING, or UNION |
| **Extra Hard** | Nested subqueries, set operations (INTERSECT, EXCEPT), or complex nesting |

---

## Evaluation Metric

Spider uses two primary metrics:

- **Exact Set Match (EM):** Normalized structural comparison of predicted SQL vs. gold SQL, ignoring value literals and clause ordering. A prediction is correct only if all SQL components (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT) match exactly.
- **Execution Accuracy (EX):** The predicted SQL is executed against the database and the result set is compared to the ground truth result. More lenient than EM as it allows semantically equivalent queries.

The official evaluation script is provided at the Spider project page.

---

## Key Design Properties

1. **Cross-domain generalization:** Databases in the train, dev, and test splits are disjoint — the model must generalize to entirely new schemas at inference time.
2. **Complex SQL coverage:** Queries span a wide range of SQL constructs: aggregation, nested subqueries, INTERSECT/EXCEPT/UNION, HAVING clauses, and multi-table JOINs.
3. **Database-split setting:** Unlike earlier benchmarks that use a single database, each database here belongs exclusively to one split, forcing models to handle schema variability.
4. **Diverse domains:** 138 different domains including entertainment, education, sports, business, government, and science.

---

## File Structure (in this repository)

```
data/spider/
├── train_spider.json       # 7,000 training samples (original)
├── train_others.json       # 1,659 training samples (adapted from other datasets)
├── dev.json                # 1,034 development samples
├── test.json               # 2,147 test samples (no gold SQL)
├── tables.json             # Schema definitions for all 166 databases
├── dev_gold.sql            # Gold SQL for dev set (one query per line)
├── train_gold.sql          # Gold SQL for training set
├── test_gold.sql           # Gold SQL for test set (held by organizers; not usable offline)
├── test_tables.json        # Schema definitions for test databases
└── database/               # SQLite files, one folder per database
    ├── concert_singer/
    │   └── concert_singer.sqlite
    └── ...
```

---

## Citation

```bibtex
@inproceedings{Yu2018Spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2018}
}
```
