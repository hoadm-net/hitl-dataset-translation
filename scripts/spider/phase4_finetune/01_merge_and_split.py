"""
Phase 4 — Spider: Merge, Deduplicate, and Split

Merges gold seed (Phase 2) and GPT translations (Phase 3). Gold seed takes
priority on ID conflicts. Applies stratified split into train/dev/test by
source (manual/gpt) and SQL hardness.

Input:
    data/spider/manual/gold_seed.json
    data/spider/gpt/translations.json

Output:
    data/spider/merged/train.json
    data/spider/merged/dev.json
    data/spider/merged/test.json
    data/spider/merged/stats.json
"""
# TODO: Implement merge and stratified split.
