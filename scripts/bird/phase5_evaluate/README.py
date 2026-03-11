"""
Phase 5 — BIRD: Downstream Text-to-SQL Evaluation

Fine-tunes Qwen2.5-Coder-7B-Instruct on English and Vietnamese BIRD data.
Evaluates on the dev set (no public test set) using:
  - Execution Accuracy (EX) — primary metric
  - Valid Efficiency Score (VES) — secondary metric
  - Breakdown by difficulty (simple / moderate / challenging)

Scripts:
    01_prepare_data.py   — build instruction-tuning JSONL (includes evidence)
    02_finetune_coder.py — fine-tune Qwen2.5-Coder (--lang en|vi)
    03_predict_sql.py    — generate SQL predictions
    04_evaluate.py       — compute EX + VES metrics

Output: results/bird/
"""
# TODO: Implement Phase 5 after Phase 4 is complete.
