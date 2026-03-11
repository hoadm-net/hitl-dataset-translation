"""
Phase 5 — Spider: Downstream Text-to-SQL Evaluation

Fine-tunes Qwen2.5-Coder-7B-Instruct separately on English and Vietnamese
Spider data, then evaluates Execution Accuracy on the test split with
breakdown by hardness level (easy / medium / hard / extra hard).

Scripts:
    01_prepare_data.py   — build instruction-tuning JSONL
    02_finetune_coder.py — fine-tune Qwen2.5-Coder (--lang en|vi)
    03_predict_sql.py    — generate SQL predictions
    04_evaluate.py       — compute EX metric, produce comparison plots

Output: results/spider/
"""
# TODO: Implement Phase 5 after Phase 4 is complete.
