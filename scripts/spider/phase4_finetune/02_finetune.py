"""
Phase 4 — Spider: Fine-tune Translation Model

Fine-tunes Qwen2.5-7B-Instruct with QLoRA on the merged dataset for the
translation task: (EN question + SQL + db_id) → VI question.

Input:  data/spider/merged/train.json
Output: models/qwen25_spider/
"""
# TODO: Implement QLoRA fine-tuning using PEFT + transformers.
