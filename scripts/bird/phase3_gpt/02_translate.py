"""
Phase 3 — BIRD: GPT Translation with Validation

Translates both `question` and `evidence` in a single API call to maintain
terminological consistency. Validation checks:
  1. LaBSE similarity for vi_question >= threshold
  2. LaBSE similarity for vi_evidence >= threshold (when evidence is non-empty)
  3. No SQL operator set mismatch

Input:
    data/bird/gpt/samples_to_translate.json
    data/bird/manual/gold_seed.json

Output:
    data/bird/gpt/translations.json
    data/bird/gpt/checkpoint.json
"""
# TODO: Implement using scripts/utils/gpt_client.py and scripts/utils/labse.py
