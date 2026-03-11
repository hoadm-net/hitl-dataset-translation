"""
Phase 2 — BIRD: Parse Label Studio Annotations

Same pipeline as Spider Phase 2, but handles the additional `evidence` field.
Both `question` and `evidence` are annotated together; the export includes
both `vi_question` and `vi_evidence` translations.

Input:
    Label Studio export JSON (path specified via --input)

Output:
    data/bird/manual/raw_annotations.json
"""
# TODO: Implement after Label Studio annotation task is set up.
