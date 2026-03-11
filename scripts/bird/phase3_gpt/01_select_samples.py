"""
Phase 3 — BIRD: Select Samples for GPT Translation

Same logic as Spider Phase 3, with an additional consideration:
each few-shot example includes both (question, vi_question) and
(evidence, vi_evidence) pairs to ensure terminological consistency
in evidence translation.

Input:
    data/bird/extracted/train.json
    data/bird/manual/gold_seed.json

Output:
    data/bird/gpt/samples_to_translate.json
"""
# TODO: Implement few-shot selection with evidence-aware pairing.
