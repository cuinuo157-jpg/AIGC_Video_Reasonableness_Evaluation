import numpy as np

from src.face_identity.csim_scorer import CSIMScorer


def _make_embeddings(n, noise=0.01):
    base = np.random.rand(512).astype(np.float32)
    base /= np.linalg.norm(base)
    embs = []
    for _ in range(n):
        e = base + np.random.randn(512).astype(np.float32) * noise
        e /= np.linalg.norm(e)
        embs.append(e)
    return embs


def test_csim_ref_high_consistency():
    embs = _make_embeddings(10, noise=0.01)
    result = CSIMScorer().compute(embs)
    assert result.csim_ref > 0.9


def test_csim_adj_high_consistency():
    result = CSIMScorer().compute(_make_embeddings(10, noise=0.01))
    assert result.csim_adj > 0.9


def test_csim_detects_drop():
    embs = _make_embeddings(10, noise=0.01)
    # Use negated base to ensure very different embedding (cosine sim ~ -1)
    embs[5] = -embs[0] + np.random.randn(512).astype(np.float32) * 0.01
    embs[5] /= np.linalg.norm(embs[5])
    result = CSIMScorer().compute(embs)
    assert len(result.drop_events) > 0
    assert result.csim_min < 0.5


def test_csim_identity_score():
    result = CSIMScorer().compute(_make_embeddings(10, noise=0.01))
    assert 0.0 <= result.identity_score <= 1.0
