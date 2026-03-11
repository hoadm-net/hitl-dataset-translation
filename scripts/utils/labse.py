"""
LaBSE-based semantic similarity for translation quality assessment.

LaBSE (Language-agnostic BERT Sentence Embedding) encodes sentences from
100+ languages into a shared embedding space, making it suitable for
cross-lingual similarity scoring between English source questions and
Vietnamese translations.

Model: sentence-transformers/LaBSE
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
MODEL_NAME = "sentence-transformers/LaBSE"


def get_model() -> SentenceTransformer:
    """Load and cache the LaBSE model (lazy initialization)."""
    global _model
    if _model is None:
        print(f"Loading LaBSE model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compute_similarity(en_text: str, vi_text: str) -> float:
    """
    Compute cosine similarity between an English text and its Vietnamese translation.

    Args:
        en_text: Source English sentence.
        vi_text: Translated Vietnamese sentence.

    Returns:
        Cosine similarity score in [0, 1].
    """
    model = get_model()
    embeddings = model.encode([en_text, vi_text], normalize_embeddings=True)
    score: float = float(np.dot(embeddings[0], embeddings[1]))
    return score


def compute_similarities_batch(
    en_texts: list[str],
    vi_texts: list[str],
    batch_size: int = 64,
) -> list[float]:
    """
    Compute cosine similarities for a batch of (EN, VI) pairs.

    Args:
        en_texts: List of English source sentences.
        vi_texts: List of Vietnamese translated sentences.
        batch_size: Encoding batch size.

    Returns:
        List of cosine similarity scores, one per pair.
    """
    assert len(en_texts) == len(vi_texts), "Lists must have equal length."
    model = get_model()

    all_texts = en_texts + vi_texts
    embeddings = model.encode(all_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    en_embs = embeddings[: len(en_texts)]
    vi_embs = embeddings[len(en_texts) :]
    scores = (en_embs * vi_embs).sum(axis=1).tolist()
    return scores
