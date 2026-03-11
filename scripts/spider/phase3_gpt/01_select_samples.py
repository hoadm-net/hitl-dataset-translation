"""
Phase 3 — Spider: Select Samples for GPT Translation

Selects the remaining (non-seed) samples from the extracted training set that
need GPT-based translation. For each target sample, retrieves k few-shot
examples from the gold seed with matching or similar SQL patterns.

Input:
    data/spider/extracted/train.json
    data/spider/manual/gold_seed.json  (from Phase 2)

Output:
    data/spider/gpt/samples_to_translate.json
"""
# TODO: Implement few-shot selection strategy (random + stratified by SQL pattern).
