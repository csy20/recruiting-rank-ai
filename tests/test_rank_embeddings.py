import numpy as np
import pytest

from rank import _load_precomputed_embeddings, _require_precomputed_embeddings


def test_load_embeddings_success(tmp_path):
    ids = ["CAND_0001", "CAND_0002", "CAND_0003"]
    embeddings = np.random.randn(3, 384).astype(np.float32)
    np.save(tmp_path / "candidate_embeddings.npy", embeddings)
    np.save(tmp_path / "candidate_id_order.npy", np.array(ids, dtype=object))

    loaded, reason = _load_precomputed_embeddings(str(tmp_path), 3, ids)

    assert reason is None
    np.testing.assert_array_equal(loaded, embeddings)


def test_load_embeddings_missing_file(tmp_path):
    loaded, reason = _load_precomputed_embeddings(str(tmp_path), 3, ["a", "b", "c"])

    assert loaded is None
    assert "not found" in reason


def test_load_embeddings_count_mismatch(tmp_path):
    ids = ["CAND_0001", "CAND_0002"]
    np.save(tmp_path / "candidate_embeddings.npy", np.random.randn(3, 384).astype(np.float32))
    np.save(tmp_path / "candidate_id_order.npy", np.array(ids, dtype=object))

    loaded, reason = _load_precomputed_embeddings(str(tmp_path), 2, ids)

    assert loaded is None
    assert "embedding count" in reason


def test_load_embeddings_id_order_mismatch(tmp_path):
    ids = ["CAND_0001", "CAND_0002"]
    np.save(tmp_path / "candidate_embeddings.npy", np.random.randn(2, 384).astype(np.float32))
    np.save(tmp_path / "candidate_id_order.npy", np.array(["CAND_0002", "CAND_0001"], dtype=object))

    loaded, reason = _load_precomputed_embeddings(str(tmp_path), 2, ids)

    assert loaded is None
    assert "order does not match" in reason


def test_require_embeddings_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit):
        _require_precomputed_embeddings(str(tmp_path), 2, ["a", "b"])