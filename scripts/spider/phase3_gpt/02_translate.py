"""
Phase 3 — Spider: GPT Translation with Validation

Translates each selected sample using GPT with few-shot prompting. Each
translation is validated in real time:
  1. LaBSE similarity >= LABSE_THRESHOLD
  2. No SQL operator set mismatch (rule-based)

Failed samples are automatically retried with a different few-shot example set.
Progress is checkpointed to allow resumption on interruption.

Input:
    data/spider/gpt/samples_to_translate.json
    data/spider/manual/gold_seed.json

Output:
    data/spider/gpt/translations.json
    data/spider/gpt/checkpoint.json  (auto-resume)
"""
# TODO: Implement using scripts/utils/gpt_client.py and scripts/utils/labse.py
