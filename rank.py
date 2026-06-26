#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    JD_KEYWORDS,
    MODEL_PATH,
    OUTPUT_PATH,
    SENTENCE_TRANSFORMER_MODEL,
)
from features.extractor import _get_combined_text, extract_all_features, load_candidates
from scoring.explainer import explain_ranking
from scoring.jd_parser import (
    get_jd_dimension_weights,
    parse_jd,
)
from scoring.ranker import (
    audit_fairness,
    calibrate_scores,
    generate_reasoning,
    load_model,
    rank_candidates,
    train_model,
)
from scoring.skill_graph import compute_concept_boost


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


_handler = _FlushHandler()
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger("rank")

_MODEL = None
_HF_CACHED = None


def _is_hf_model_cached(model_name: str) -> bool:
    global _HF_CACHED
    if _HF_CACHED is not None:
        return _HF_CACHED
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = model_name.replace("/", "--")
    _HF_CACHED = os.path.isdir(os.path.join(cache_dir, f"models--{model_dir}"))
    return _HF_CACHED


def _get_sentence_transformer():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        if _is_hf_model_cached(SENTENCE_TRANSFORMER_MODEL):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logger.info("Loading sentence-transformer model: %s", SENTENCE_TRANSFORMER_MODEL)
        _MODEL = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _MODEL


def _build_jd_text() -> str:
    terms: set = set()
    for dim in JD_KEYWORDS.values():
        for term in dim.get("terms", []):
            terms.add(term)
    return " ".join(sorted(terms))


def _load_jd_text(jd_path: str) -> str | None:
    if not jd_path:
        return None
    if not os.path.exists(jd_path):
        logger.warning("JD file %s not found, falling back to keyword-derived JD", jd_path)
        return None
    try:
        with open(jd_path) as f:
            text = f.read().strip()
        if text:
            logger.info("Loaded JD text from %s (%d chars)", jd_path, len(text))
            return text
    except OSError as e:
        logger.warning("Failed to read JD file %s: %s", jd_path, e)
    return None


def _get_candidate_text(candidate: dict[str, Any]) -> str:
    return _get_combined_text(candidate)


def _encode_with_chunking(
    texts: list[str],
    model: Any,
    normalize: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    n = len(texts)
    logger.info("Encoding %d texts (batch_size=%d)...", n, batch_size)
    if n == 0:
        return np.array([])

    model_max_len = getattr(model, "max_seq_length", None) or getattr(
        model.tokenizer, "model_max_length", 512
    )
    if not isinstance(model_max_len, int) or model_max_len > 10000:
        model_max_len = 512
    max_content_len = max(1, model_max_len - 2)

    tokenized = [
        model.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        for text in texts
    ]
    short_idx = [i for i, toks in enumerate(tokenized) if len(toks) <= max_content_len]
    long_idx = [i for i, toks in enumerate(tokenized) if len(toks) > max_content_len]
    result: list[np.ndarray | None] = [None] * n

    t_enc = time.time()
    n_short = len(short_idx)
    n_long = len(long_idx)
    logger.info(
        "  short texts: %d, long texts: %d (max_seq_len=%d)", n_short, n_long, max_content_len
    )

    for start in range(0, n_short, batch_size):
        indices = short_idx[start : start + batch_size]
        batch = [texts[i] for i in indices]
        embs = model.encode(
            batch, show_progress_bar=False, normalize_embeddings=normalize, batch_size=batch_size
        )
        for idx, emb in zip(indices, embs, strict=True):
            result[idx] = emb
        batch_num = start // batch_size
        if batch_num > 0 and batch_num % 20 == 0:
            done = min(start + batch_size, n_short)
            elapsed = time.time() - t_enc
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_short - done) / rate if rate > 0 else 0
            logger.info(
                "  short texts: %d/%d encoded (%.1fs, ~%.1fs remaining)",
                done,
                n_short,
                elapsed,
                remaining,
            )

    t_long = time.time()
    for li, idx in enumerate(long_idx):
        tokens = tokenized[idx]
        chunks = [
            model.tokenizer.decode(
                tokens[start : start + max_content_len],
                skip_special_tokens=True,
            )
            for start in range(0, len(tokens), max_content_len)
        ]
        chunk_batches: list[np.ndarray] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            chunk_batches.append(
                model.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                    batch_size=batch_size,
                )
            )
        chunk_embs = np.concatenate(chunk_batches, axis=0)
        pooled = np.mean(chunk_embs, axis=0)
        if normalize:
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
        result[idx] = pooled
        if n_long > 1 and (li + 1) % 100 == 0:
            elapsed = time.time() - t_long
            rate = (li + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_long - li - 1) / rate if rate > 0 else 0
            logger.info(
                "  long texts: %d/%d chunked (%.1fs, ~%.1fs remaining)",
                li + 1,
                n_long,
                elapsed,
                remaining,
            )

    encoded = [emb for emb in result if emb is not None]
    return np.array(encoded)


def _hash_candidate_ids(candidates: list[dict[str, Any]]) -> str:
    ids_concat = "\x00".join(
        c.get("candidate_id") or f"__idx_{i}__" for i, c in enumerate(candidates)
    )
    return hashlib.sha256(ids_concat.encode("utf-8")).hexdigest()


def _checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, "embedding_checkpoint.json")


def _partial_emb_path(out_dir: str) -> str:
    return os.path.join(out_dir, "candidate_embeddings.partial.npy")


def _partial_order_path(out_dir: str) -> str:
    return os.path.join(out_dir, "candidate_id_order.partial.npy")


def _read_embedding_checkpoint(out_dir: str, total_count: int, candidates_hash: str) -> int | None:
    ckpt = _checkpoint_path(out_dir)
    if not os.path.exists(ckpt):
        return None
    try:
        with open(ckpt) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupted checkpoint %s, starting fresh", ckpt)
        return None

    if data.get("total_count") != total_count:
        logger.info(
            "Checkpoint discarded: candidate count changed (%s -> %s)",
            data.get("total_count"),
            total_count,
        )
        return None
    if data.get("candidates_hash") != candidates_hash:
        logger.info("Checkpoint discarded: candidate list hash changed")
        return None
    if data.get("model") != SENTENCE_TRANSFORMER_MODEL:
        logger.info(
            "Checkpoint discarded: model changed (%s -> %s)",
            data.get("model"),
            SENTENCE_TRANSFORMER_MODEL,
        )
        return None

    last_index = data.get("last_index", 0)
    if not isinstance(last_index, int) or last_index < 0 or last_index > total_count:
        logger.warning("Invalid last_index %s in checkpoint, starting fresh", last_index)
        return None

    if not os.path.exists(_partial_emb_path(out_dir)) or not os.path.exists(
        _partial_order_path(out_dir)
    ):
        logger.warning(
            "Partial embedding files missing (checkpoint exists but data gone), starting fresh"
        )
        return None

    logger.info("Resuming embedding encode from checkpoint at index %d/%d", last_index, total_count)
    return last_index


def _write_embedding_checkpoint(
    out_dir: str, last_index: int, total_count: int, candidates_hash: str
) -> None:
    with open(_checkpoint_path(out_dir), "w") as f:
        json.dump(
            {
                "last_index": last_index,
                "total_count": total_count,
                "candidates_hash": candidates_hash,
                "model": SENTENCE_TRANSFORMER_MODEL,
            },
            f,
        )


def _precompute_embeddings(
    candidates: list[dict[str, Any]],
    out_dir: str,
    force: bool = False,
) -> np.ndarray:
    n = len(candidates)
    if n == 0:
        return np.array([])

    texts = ["passage: " + _get_candidate_text(c) for c in candidates]
    logger.info(
        "Encoding %d candidates with sentence-transformers (%s)...",
        n,
        SENTENCE_TRANSFORMER_MODEL,
    )

    candidates_hash = _hash_candidate_ids(candidates)

    start_idx = 0
    if not force:
        ckpt_idx = _read_embedding_checkpoint(out_dir, n, candidates_hash)
        if ckpt_idx is not None:
            start_idx = ckpt_idx

    if start_idx > 0 and start_idx >= n:
        logger.info("Embeddings already fully encoded per checkpoint, finalizing...")
        return np.load(os.path.join(out_dir, "candidate_embeddings.npy"))

    cid_order = np.array(
        [c.get("candidate_id") or str(i) for i, c in enumerate(candidates)],
        dtype=object,
    )

    t0 = time.time()
    model = _get_sentence_transformer()

    model_max_len = getattr(model, "max_seq_length", None) or getattr(
        model.tokenizer, "model_max_length", 512
    )
    if not isinstance(model_max_len, int) or model_max_len > 10000:
        model_max_len = 512
    max_content_len = max(1, model_max_len - 2)

    logger.info("Tokenizing %d texts to classify short/long...", n)
    t_tokenize = time.time()
    is_long = [False] * n
    long_tokens: list[list[int] | None] = [None] * n
    tokenize_batch_size = 10000
    for batch_start in range(0, n, tokenize_batch_size):
        batch_end = min(batch_start + tokenize_batch_size, n)
        encoded = model.tokenizer(
            texts[batch_start:batch_end],
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        for i_local, input_ids in enumerate(encoded["input_ids"]):
            i = batch_start + i_local
            if len(input_ids) > max_content_len:
                is_long[i] = True
                long_tokens[i] = input_ids
    n_long = sum(is_long)
    logger.info(
        "Tokenization done in %.1fs — short: %d, long: %d",
        time.time() - t_tokenize,
        n - n_long,
        n_long,
    )

    partial_emb = _partial_emb_path(out_dir)
    partial_order = _partial_order_path(out_dir)

    resume_indices: np.ndarray | None = None
    if start_idx == 0 and not force and os.path.exists(partial_emb):
        try:
            tmp = np.load(partial_emb, mmap_mode="r")
            if tmp.ndim == 2 and tmp.shape[0] == n:
                zero_mask = np.all(tmp == 0, axis=1)
                zero_count = int(zero_mask.sum())
                if 0 < zero_count < n:
                    logger.info(
                        "Existing partial embeddings found (%d/%d done), "
                        "resuming for %d missing rows",
                        n - zero_count,
                        n,
                        zero_count,
                    )
                    resume_indices = np.where(zero_mask)[0]
            del tmp
        except Exception:
            pass

    if resume_indices is not None:
        tmp = np.load(partial_emb, mmap_mode="r")
        emb_dim = tmp.shape[1]
        del tmp
        emb_memmap = np.lib.format.open_memmap(
            partial_emb,
            mode="r+",
            dtype=np.float32,
            shape=(n, emb_dim),
        )
        if not os.path.exists(partial_order):
            np.save(partial_order, cid_order)
    elif start_idx == 0:
        for p in [partial_emb, partial_order, _checkpoint_path(out_dir)]:
            if os.path.exists(p):
                os.remove(p)

        first_emb = model.encode(
            texts[:1],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb_dim = first_emb.shape[1]

        emb_memmap = np.lib.format.open_memmap(
            partial_emb,
            mode="w+",
            dtype=np.float32,
            shape=(n, emb_dim),
        )
        np.save(partial_order, cid_order)
    else:
        tmp = np.load(partial_emb, mmap_mode="r")
        emb_dim = tmp.shape[1]
        del tmp

        emb_memmap = np.lib.format.open_memmap(
            partial_emb,
            mode="r+",
            dtype=np.float32,
            shape=(n, emb_dim),
        )

    par_batch_size = 64

    if resume_indices is not None:
        missing = resume_indices.tolist()
        missing_n = len(missing)
        logger.info("Sequential resume: encoding %d missing rows...", missing_n)
        missing_short = [i for i in missing if not is_long[i]]
        missing_long = [i for i in missing if is_long[i]]
        t_enc = time.time()
        completed = 0
        for start in range(0, len(missing_short), par_batch_size):
            chunk = missing_short[start : start + par_batch_size]
            embs = model.encode(
                [texts[i] for i in chunk],
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=len(chunk),
            )
            for idx, emb in zip(chunk, embs, strict=True):
                emb_memmap[idx] = emb
            completed += len(chunk)
            elapsed = time.time() - t_enc
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (missing_n - completed) / rate if rate > 0 else 0
            if completed % 5000 == 0 or completed == missing_n:
                logger.info(
                    "  resume: %d/%d short encoded (%.1f%%, %.1fs elapsed, ~%.1fs remaining)",
                    completed,
                    missing_n,
                    100.0 * completed / missing_n,
                    elapsed,
                    remaining,
                )
        # Batch ALL chunk texts from ALL remaining long texts together
        all_chunk_texts: list[str] = []
        all_chunk_owners: list[int] = []  # candidate index for each chunk
        for idx in missing_long:
            tokens = long_tokens[idx]
            if not tokens:
                continue
            chunks = [
                model.tokenizer.decode(tokens[s : s + max_content_len], skip_special_tokens=True)
                for s in range(0, len(tokens), max_content_len)
            ]
            for ct in chunks:
                all_chunk_texts.append(ct)
                all_chunk_owners.append(idx)

        n_chunks = len(all_chunk_texts)
        logger.info(
            "  resume long: %d candidates, %d total chunks, encoding in batches...",
            len(missing_long),
            n_chunks,
        )
        enc_batch = 256
        all_chunk_embs = np.zeros((n_chunks, emb_dim), dtype=np.float32)
        for start in range(0, n_chunks, enc_batch):
            end = min(start + enc_batch, n_chunks)
            batch_embs = model.encode(
                all_chunk_texts[start:end],
                show_progress_bar=False,
                normalize_embeddings=False,
                batch_size=128,
            )
            all_chunk_embs[start:end] = batch_embs

        # Group chunk embeddings by candidate and mean-pool
        import collections

        candidate_to_embs: dict[int, list[np.ndarray]] = collections.defaultdict(list)
        for ci, owner in enumerate(all_chunk_owners):
            candidate_to_embs[owner].append(all_chunk_embs[ci])
        for idx, chunk_list in candidate_to_embs.items():
            pooled = np.mean(np.stack(chunk_list), axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            emb_memmap[idx] = pooled

        completed += len(missing_long)
        logger.info(
            "  resume long: %d done (%.1fs elapsed)",
            len(missing_long),
            time.time() - t_enc,
        )
        del emb_memmap
        logger.info("Resume encoding done in %.1fs", time.time() - t_enc)
    else:
        seq_batch_size = 256
        total_batches = (n + seq_batch_size - 1) // seq_batch_size
        resume_batch = start_idx // seq_batch_size
        log_every = max(1, total_batches // 20)

        for batch_num in range(resume_batch, total_batches):
            batch_start = batch_num * seq_batch_size
            batch_end = min(batch_start + seq_batch_size, n)
            batch_texts = texts[batch_start:batch_end]
            batch_long = is_long[batch_start:batch_end]
            batch_count = batch_end - batch_start

            batch_result = np.zeros((batch_count, emb_dim), dtype=np.float32)

            short_in_batch = [i for i, l in enumerate(batch_long) if not l]
            if short_in_batch:
                short_texts = [batch_texts[i] for i in short_in_batch]
                short_embs = model.encode(
                    short_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=128,
                )
                for iib, emb in zip(short_in_batch, short_embs):
                    batch_result[iib] = emb

            long_in_batch = [i for i, l in enumerate(batch_long) if l]
            for iib in long_in_batch:
                global_idx = batch_start + iib
                tokens = long_tokens[global_idx]
                if tokens is None:
                    tokens = model.tokenizer.encode(
                        texts[global_idx],
                        add_special_tokens=False,
                        truncation=False,
                        verbose=False,
                    )
                chunks = [
                    model.tokenizer.decode(
                        tokens[s : s + max_content_len],
                        skip_special_tokens=True,
                    )
                    for s in range(0, len(tokens), max_content_len)
                ]
                chunk_embs_list = []
                for chunk_start in range(0, len(chunks), seq_batch_size):
                    chunk_batch = chunks[chunk_start : chunk_start + seq_batch_size]
                    chunk_embs_list.append(
                        model.encode(
                            chunk_batch,
                            show_progress_bar=False,
                            normalize_embeddings=False,
                            batch_size=128,
                        )
                    )
                if chunk_embs_list:
                    chunk_embs = np.concatenate(chunk_embs_list, axis=0)
                    pooled = np.mean(chunk_embs, axis=0)
                    norm = np.linalg.norm(pooled)
                    if norm > 0:
                        pooled = pooled / norm
                    batch_result[iib] = pooled

            emb_memmap[batch_start:batch_end] = batch_result
            _write_embedding_checkpoint(out_dir, batch_end, n, candidates_hash)

            if (batch_num + 1) % log_every == 0 or batch_end == n:
                elapsed = time.time() - t0
                rate = batch_end / elapsed if elapsed > 0 else 0
                remaining = (n - batch_end) / rate if rate > 0 else 0
                pct = 100.0 * batch_end / n
                logger.info(
                    "  embeddings: %d/%d encoded (%.1f%%, %.1fs elapsed, ~%.1fs remaining)",
                    batch_end,
                    n,
                    pct,
                    elapsed,
                    remaining,
                )

        del emb_memmap

    final_emb = os.path.join(out_dir, "candidate_embeddings.npy")
    final_order = os.path.join(out_dir, "candidate_id_order.npy")
    os.rename(partial_emb, final_emb)
    os.rename(partial_order, final_order)
    logger.info("Saved embeddings to %s", final_emb)
    logger.info("Saved candidate ID order to %s", final_order)
    logger.info(
        "Encoded %d embeddings in %.1fs, shape=%s",
        n,
        time.time() - t0,
        np.load(final_emb, mmap_mode="r").shape,
    )

    ckpt = _checkpoint_path(out_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    emb_ids_path = os.path.join(out_dir, "embedding_ids.csv")
    with open(emb_ids_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "embedding_index"])
        for i, c in enumerate(candidates):
            w.writerow([c.get("candidate_id") or str(i), i])
    logger.info("Saved embedding IDs to %s (%d entries)", emb_ids_path, n)

    return np.load(final_emb)


def _build_bm25_index(candidates: list[dict[str, Any]]) -> Any:
    from rank_bm25 import BM25Okapi

    texts = [_get_candidate_text(c) for c in candidates]
    tokenized = [t.lower().split() for t in texts]
    logger.info("Building BM25 index on %d candidates...", len(tokenized))
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built in %.1fs", time.time() - t0)
    return bm25


def _compute_semantic_features(
    candidates: list[dict[str, Any]],
    all_texts: list[str],
    jd_text: str,
    precomputed_dir: str | None = None,
    orig_indices: list[int] | None = None,
) -> list[float]:
    logger.info("Computing semantic similarity features...")

    if precomputed_dir and orig_indices is not None:
        candidate_embs, reason = _load_precomputed_embeddings(
            precomputed_dir,
            len(candidates),
            [c.get("candidate_id") or str(i) for i, c in enumerate(candidates)],
        )
        if candidate_embs is not None:
            emb_path = os.path.join(precomputed_dir, "candidate_embeddings.npy")
            logger.info(
                "Loaded precomputed embeddings for %d candidates from %s (shape=%s)",
                len(candidates),
                emb_path,
                candidate_embs.shape,
            )
            model = _get_sentence_transformer()
            jd_emb = model.encode([jd_text], show_progress_bar=False, normalize_embeddings=True)
            selected = candidate_embs[orig_indices]
            sims = cosine_similarity(jd_emb, selected).flatten()
            return [float(s) for s in sims]
        logger.warning(
            "Precomputed embeddings not used: %s. Falling back to live encoding.",
            reason,
        )

    model = _get_sentence_transformer()
    jd_emb = model.encode([jd_text], show_progress_bar=False, normalize_embeddings=True)
    candidate_embs = _encode_with_chunking(all_texts, model)
    similarities = cosine_similarity(candidate_embs, jd_emb).flatten()
    return [float(s) for s in similarities]


def _extract_single(
    args: tuple[dict[str, Any], int],
) -> tuple[str, dict[str, float]]:
    candidate, idx = args
    cid = candidate.get("candidate_id") or str(idx)
    return cid, extract_all_features(candidate)


def _extract_batch(
    batch: list[tuple[dict[str, Any], int]],
) -> list[tuple[str, dict[str, float]]]:
    return [_extract_single(args) for args in batch]


def _extract_all(
    candidates: list[dict[str, Any]],
) -> list[tuple[str, dict[str, float]]]:
    n = len(candidates)
    if n == 0:
        return []

    if n < 500:
        results: list[tuple[str, dict[str, float]]] = []
        for i, c in enumerate(candidates):
            if i % 100 == 0 and i > 0:
                logger.info("  processing %d/%d...", i, n)
            cid = c.get("candidate_id") or str(i)
            results.append((cid, extract_all_features(c)))
        return results

    batch_size = 1000
    n_workers = min(8, os.cpu_count() or 1)
    logger.info("Using multiprocessing (%d workers, batch_size=%d)...", n_workers, batch_size)
    args = [(c, i) for i, c in enumerate(candidates)]
    batches = [args[i : i + batch_size] for i in range(0, len(args), batch_size)]
    results_map: dict[int, tuple[str, dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_batch, b): b[0][1] for b in batches}
        for future in as_completed(futures):
            start_idx = futures[future]
            try:
                for cid, feats in future.result():
                    results_map[start_idx] = (cid, feats)
                    start_idx += 1
            except Exception as e:
                logger.error("Batch failed starting at index %d: %s", start_idx, e)
                for i in range(start_idx, min(start_idx + batch_size, n)):
                    cid = candidates[i].get("candidate_id") or str(i)
                    results_map[i] = (cid, {})
    return [results_map[i] for i in range(n)]


def _load_precomputed_embeddings(
    precomputed_dir: str | None,
    expected_count: int,
    expected_ids: list[str] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    if not precomputed_dir:
        return None, "no --features-dir specified"

    emb_path = os.path.join(precomputed_dir, "candidate_embeddings.npy")
    if not os.path.exists(emb_path):
        return None, f"candidate_embeddings.npy not found at {emb_path}"

    try:
        embeddings = np.load(emb_path)
    except Exception as exc:
        return None, f"failed to load {emb_path}: {type(exc).__name__}: {exc}"

    if embeddings.ndim != 2:
        return None, f"invalid embedding shape {embeddings.shape} in {emb_path}"

    if embeddings.shape[0] != expected_count:
        return None, (
            f"embedding count ({embeddings.shape[0]}) != candidate count "
            f"({expected_count}) in {emb_path}"
        )

    id_order_path = os.path.join(precomputed_dir, "candidate_id_order.npy")
    if os.path.exists(id_order_path):
        try:
            stored_ids = np.load(id_order_path, allow_pickle=True)
        except Exception as exc:
            return None, f"failed to load {id_order_path}: {type(exc).__name__}: {exc}"

        if len(stored_ids) != expected_count:
            return None, (
                f"candidate_id_order.npy has {len(stored_ids)} IDs but expected {expected_count}"
            )

        if expected_ids is not None and list(stored_ids) != expected_ids:
            if set(stored_ids) != set(expected_ids):
                return None, "candidate_id_order.npy IDs do not match current candidates"
            return None, "candidate_id_order.npy order does not match current candidates"
    elif expected_ids is not None:
        return None, (
            f"candidate_id_order.npy not found at {id_order_path} "
            "(cannot verify embedding alignment)"
        )

    return embeddings, None


def _require_precomputed_embeddings(
    precomputed_dir: str | None,
    expected_count: int,
    expected_ids: list[str],
) -> np.ndarray:
    embeddings, reason = _load_precomputed_embeddings(
        precomputed_dir,
        expected_count,
        expected_ids,
    )
    if embeddings is not None:
        emb_path = os.path.join(precomputed_dir, "candidate_embeddings.npy")
        logger.info(
            "Loaded precomputed embeddings for %d candidates from %s (shape=%s)",
            expected_count,
            emb_path,
            embeddings.shape,
        )
        return embeddings

    logger.error(
        "Precomputed embeddings not available: %s. "
        "Cannot rank candidates without precomputed embeddings. "
        "Run 'python rank.py --precompute ...' first.",
        reason,
    )
    raise SystemExit(1)


def _candidate_count(out_dir: str) -> int | None:
    meta_path = os.path.join(out_dir, "metadata.csv")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return sum(1 for _ in f) - 1
    return None


def precompute(
    path: str,
    out_dir: str,
    jd_path: str | None = None,
    train: bool = False,
    force: bool = False,
):
    t0 = time.time()
    logger.info("Loading candidates from %s...", path)
    candidates = load_candidates(path)
    if not candidates:
        logger.warning("No candidates loaded!")
        return
    n_candidates = len(candidates)
    logger.info("Loaded %d candidates in %.1fs", n_candidates, time.time() - t0)

    os.makedirs(out_dir, exist_ok=True)

    ids: list[str] = []
    all_features: list[dict[str, float]] = []

    npz_path = os.path.join(out_dir, "features.npz")
    existing_count = _candidate_count(out_dir)
    features_exist = existing_count is not None and existing_count == n_candidates

    if features_exist and not force:
        logger.info(
            "features.npz already exists with %d candidates, skipping extraction (use --force to redo)",
            existing_count,
        )
        npz = np.load(npz_path)
        feature_matrix = npz["features"]
        feature_keys = list(npz["keys"])
    else:
        logger.info("Extracting features...")
        extracted = _extract_all(candidates)
        for cid, feats in extracted:
            ids.append(cid)
            all_features.append(feats)

        if not all_features:
            logger.warning("No features extracted!")
            return

        feature_keys = list(all_features[0].keys())
        feature_matrix = np.zeros((n_candidates, len(feature_keys)), dtype=np.float32)
        for i, feats in enumerate(all_features):
            for j, key in enumerate(feature_keys):
                val = feats.get(key, 0.0)
                if val is None:
                    val = 0.0
                feature_matrix[i, j] = float(val)

        np.savez_compressed(npz_path, features=feature_matrix, keys=feature_keys)
        logger.info("Saved features to %s", npz_path)

        meta_path = os.path.join(out_dir, "metadata.csv")
        with open(meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "feature_index"])
            for i, cid in enumerate(ids):
                w.writerow([cid, i])
        logger.info("Saved metadata to %s", meta_path)

    emb_path = os.path.join(out_dir, "candidate_embeddings.npy")
    id_order_path = os.path.join(out_dir, "candidate_id_order.npy")
    embeddings_exist = os.path.exists(emb_path) and os.path.exists(id_order_path)

    if embeddings_exist and not force:
        logger.info(
            "candidate_embeddings.npy and candidate_id_order.npy already exist, "
            "skipping encoding (use --force to redo)"
        )
    else:
        _precompute_embeddings(candidates, out_dir, force=force)

    if train:
        model_save_path = os.path.join(out_dir, "ranker_model.pkl")
        if os.path.exists(model_save_path) and not force:
            logger.info("ranker_model.pkl already exists, skipping training (use --force to redo)")
        else:
            training_features = all_features
            if not training_features and features_exist:
                logger.info("Reconstructing feature dicts from features.npz for training...")
                training_features = [
                    dict(zip(feature_keys, feature_matrix[i].tolist())) for i in range(n_candidates)
                ]
            if training_features:
                logger.info("Training ML model on behavioral signals...")
                t_train = time.time()
                try:
                    signals_list = [c.get("redrob_signals", {}) for c in candidates]
                    train_model(
                        training_features, signals_list=signals_list, model_path=model_save_path
                    )
                except Exception:
                    logger.exception("Training failed")
                    raise
                logger.info("Training complete in %.1fs", time.time() - t_train)

    artifacts: list[tuple[str, str]] = [
        ("features.npz", "Feature matrix"),
        ("metadata.csv", "Feature index"),
        ("candidate_embeddings.npy", "Candidate embeddings"),
        ("candidate_id_order.npy", "Candidate ID order"),
        ("embedding_ids.csv", "Embedding-to-ID mapping"),
        ("ranker_model.pkl", "Trained ML model"),
    ]
    logger.info("=" * 60)
    logger.info("ARTIFACT SUMMARY (dir: %s)", out_dir)
    logger.info("=" * 60)
    for fname, desc in artifacts:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(fpath)))
            logger.info("  [OK]   %-30s %s  (%7.1f KB, %s)", fname, desc, size_kb, mtime)
        else:
            logger.info("  [MISS] %-30s %s  (not found)", fname, desc)
    logger.info("=" * 60)

    logger.info("Precomputation done in %.1fs total", time.time() - t0)


def rank(
    candidates_path: str,
    output_path: str,
    precomputed_dir: str | None = None,
    jd_path: str | None = None,
    model_path: str | None = None,
    output_json: bool = False,
):
    t0 = time.time()
    logger.info("Loading candidates from %s...", candidates_path)
    candidates = load_candidates(candidates_path)
    if not candidates:
        logger.warning("No candidates loaded!")
        return
    n_total = len(candidates)
    logger.info("Loaded %d candidates in %.1fs", n_total, time.time() - t0)

    features_dir = os.path.abspath(precomputed_dir or "data")
    logger.info("Using features directory: %s", features_dir)
    precomputed_dir = features_dir

    candidate_ids = [c.get("candidate_id") or str(i) for i, c in enumerate(candidates)]

    jd_profile = None
    jd_text_loaded = _load_jd_text(jd_path) if jd_path else None
    if jd_text_loaded:
        jd_profile = parse_jd(jd_text_loaded)
        jd_weights = get_jd_dimension_weights(jd_profile)
        logger.info("JD parsed: %d dimensions analyzed", len(jd_weights))
    else:
        jd_weights = None

    ref_text = jd_text_loaded or _build_jd_text()

    # ====================================================================
    # STAGE 1 — BM25 scoring over ALL candidates (no prefilter)
    # ====================================================================
    stage_times: dict[str, float] = {}
    t1 = time.time()
    bm25 = _build_bm25_index(candidates)
    tokenized_query = ref_text.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_ranks = scipy.stats.rankdata(-bm25_scores, method="average")
    stage_times["bm25_full"] = time.time() - t1
    logger.info("STAGE 1 — BM25 scored %d candidates in %.2fs", n_total, stage_times["bm25_full"])

    # ====================================================================
    # STAGE 2 — Dense cosine similarity over ALL candidates
    # ====================================================================
    t2 = time.time()
    model = _get_sentence_transformer()
    jd_query = "query: " + ref_text
    jd_emb = model.encode([jd_query], show_progress_bar=False, normalize_embeddings=True)

    candidate_embs = _require_precomputed_embeddings(precomputed_dir, n_total, candidate_ids)

    dense_sims = cosine_similarity(jd_emb, candidate_embs).flatten()
    dense_ranks = scipy.stats.rankdata(-dense_sims, method="average")
    stage_times["dense_full"] = time.time() - t2
    logger.info("STAGE 2 — Dense scored %d candidates in %.2fs", n_total, stage_times["dense_full"])

    # ====================================================================
    # STAGE 3 — RRF fusion (Reciprocal Rank Fusion, k=60)
    # ====================================================================
    t3 = time.time()
    k = 60
    rrf_scores = 1.0 / (k + bm25_ranks) + 1.0 / (k + dense_ranks)
    rrf_min, rrf_max = float(rrf_scores.min()), float(rrf_scores.max())
    if rrf_max > rrf_min:
        rrf_norm = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
    else:
        rrf_norm = np.zeros_like(rrf_scores)
    stage_times["rrf_fusion"] = time.time() - t3
    logger.info(
        "STAGE 3 — RRF fusion (k=%d) in %.2fs (norm range: %.4f-%.4f)",
        k,
        stage_times["rrf_fusion"],
        float(rrf_norm.min()),
        float(rrf_norm.max()),
    )

    # ====================================================================
    # STAGE 4 — Concept graph boost
    # ====================================================================
    t4 = time.time()
    concept_boosts = np.array(
        [compute_concept_boost(ref_text, _get_candidate_text(c)) for c in candidates],
        dtype=np.float32,
    )
    stage_times["concept_boost"] = time.time() - t4
    logger.info("STAGE 4 — Concept boost computed in %.2fs", stage_times["concept_boost"])

    # ====================================================================
    # STAGE 5 — Combined semantic score + select top-K
    # ====================================================================
    semantic_match = rrf_norm + concept_boosts
    top_k = min(2000, n_total)
    top_indices = np.argsort(-semantic_match)[:top_k]
    used_indices: list[int] = list(top_indices)
    logger.info(
        "STAGE 5 — Selected top %d/%d by RRF+concept fusion",
        top_k,
        n_total,
    )

    # ====================================================================
    # STAGE 6 — 7-dim feature extraction + scoring on top-K subset
    # ====================================================================
    t6 = time.time()
    precomputed = precomputed_dir and os.path.exists(os.path.join(precomputed_dir, "features.npz"))
    all_features: list[dict[str, float]] = []

    if precomputed:
        npz = np.load(os.path.join(precomputed_dir, "features.npz"))
        feature_matrix = npz["features"]
        feature_keys = list(npz["keys"])

        meta_path = os.path.join(precomputed_dir, "metadata.csv")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                reader = csv.DictReader(f)
                id_to_idx = {row["candidate_id"]: int(row["feature_index"]) for row in reader}

            for orig_idx in used_indices:
                c = candidates[orig_idx]
                cid = c.get("candidate_id")
                if cid in id_to_idx:
                    idx = id_to_idx[cid]
                    feats = {
                        feature_keys[j]: float(feature_matrix[idx, j])
                        for j in range(len(feature_keys))
                    }
                else:
                    feats = extract_all_features(c)
                feats["semantic_similarity"] = float(semantic_match[orig_idx])
                all_features.append(feats)
        else:
            for orig_idx in used_indices:
                feats = extract_all_features(candidates[orig_idx])
                feats["semantic_similarity"] = float(semantic_match[orig_idx])
                all_features.append(feats)
    else:
        logger.info("Extracting features for top %d on the fly...", len(used_indices))
        top_candidates = [candidates[i] for i in used_indices]
        extracted = _extract_all(top_candidates)
        for i, (_cid, feats) in enumerate(extracted):
            feats["semantic_similarity"] = float(semantic_match[used_indices[i]])
            all_features.append(feats)

    stage_times["scoring_topk"] = time.time() - t6
    logger.info(
        "STAGE 6 — 7-dim scoring prepped for %d candidates in %.2fs",
        len(all_features),
        stage_times["scoring_topk"],
    )

    if not all_features:
        logger.warning("No features available for ranking!")
        return

    # ====================================================================
    # STAGE 7 — ML model, ranking, calibration, reasoning, output
    # ====================================================================
    used_candidates = [candidates[i] for i in used_indices]

    model_path = model_path or MODEL_PATH
    ml_model = load_model(model_path) if os.path.exists(model_path) else None
    if ml_model is None:
        logger.info("No pre-trained model found, training from behavioral signals...")
        sigs = [c.get("redrob_signals", {}) for c in used_candidates]
        ml_model = train_model(all_features, signals_list=sigs, model_path=model_path)

    logger.info("Ranking candidates...")
    ids = [c.get("candidate_id") or str(i) for i, c in enumerate(used_candidates)]
    sigs = [c.get("redrob_signals", {}) for c in used_candidates]
    ranked = rank_candidates(
        ids,
        all_features,
        jd_weights=jd_weights,
        ml_model=ml_model,
        signals_list=sigs,
    )

    logger.info("Calibrating scores...")
    ranked = calibrate_scores(ranked)

    logger.info("Generating reasoning...")
    id_to_feats: dict[str, dict[str, float]] = dict(zip(ids, all_features, strict=True))
    id_to_candidate: dict[str, dict[str, Any]] = {
        c.get("candidate_id") or str(i): c for i, c in enumerate(used_candidates)
    }
    results: list[tuple[str, int, float, str, dict[str, float]]] = []
    for cid, score, rank_pos, dims in ranked[:100]:
        feats = id_to_feats.get(cid, {}) or {}
        cand = id_to_candidate.get(cid, {})
        reasoning = generate_reasoning(cid, score, rank_pos, dims, feats, candidate=cand)
        results.append((cid, rank_pos, round(score / 100.0, 4), reasoning, dims))

    if output_json:
        json_output = []
        for cid, rank_pos, score, reasoning, dims in results:
            feats = id_to_feats.get(cid, {}) or {}
            explanation = explain_ranking(feats, dims, score)
            entry = {
                "candidate_id": cid,
                "rank": rank_pos,
                "score": score,
                "reasoning": reasoning,
                "dimensions": {k: round(float(v), 4) for k, v in dims.items() if v is not None},
                "feature_contributions": explanation["feature_contributions"],
                "strengths": explanation["strengths"],
                "weaknesses": explanation["weaknesses"],
            }
            json_output.append(entry)

        json_path = (
            output_path.replace(".csv", ".json")
            if output_path.endswith(".csv")
            else (output_path + ".json")
        )
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        logger.info("Written JSON to %s", json_path)

        audit = audit_fairness(ranked[:100], all_features, ids)
        if audit.get("audit_enabled"):
            audit_path = (
                output_path.replace(".csv", "_fairness_audit.json")
                if output_path.endswith(".csv")
                else (output_path + "_fairness_audit.json")
            )
            with open(audit_path, "w") as f:
                json.dump(audit, f, indent=2)
            flagged = []
            for group, dims in audit.get("disparities", {}).items():
                for dim, info in dims.items():
                    if info.get("flagged"):
                        flagged.append(f"{group}/{dim}")
            if flagged:
                logger.warning("Fairness flags: %s", ", ".join(flagged))
            else:
                logger.info("Fairness audit passed — no significant disparities detected")

    logger.info("Writing top-100 to %s...", output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank_pos, score, reasoning, _dims in results:
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    # Performance report
    elapsed = time.time() - t0
    total_stages = sum(stage_times.values())
    logger.info("=" * 60)
    logger.info("PERFORMANCE TIMING REPORT")
    logger.info("=" * 60)
    for stage_name, stage_dur in sorted(stage_times.items()):
        logger.info("  %-20s: %7.2fs", stage_name, stage_dur)
    logger.info("  %-20s: %7.2fs", "stages_total", total_stages)
    logger.info("  %-20s: %7.2fs", "wall_clock", elapsed)
    logger.info("=" * 60)
    logger.info("Done in %.1fs", elapsed)


def serve(host: str, port: int, model_path: str | None = None):
    import serve as serve_module

    serve_module.main(host=host, port=port, model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description="Recruiting Rank AI — Candidate Ranking System")
    parser.add_argument("--candidates", help="Path to candidates.jsonl")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output CSV path")
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Precompute features (run before ranking on large datasets)",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Directory with precomputed features (data/ by default)",
    )
    parser.add_argument(
        "--jd",
        default=None,
        help="Path to JD description text file for similarity scoring",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train ML model from rule-based scores during precompute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing artifacts (skip checkpoint/resume)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to pre-trained ML model (.pkl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON results + fairness audit alongside CSV",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server for real-time ranking",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")

    args = parser.parse_args()

    if args.serve:
        serve(args.host, args.port, args.model)
        return

    if not args.candidates:
        parser.error("--candidates is required (or use --serve for API mode)")

    if args.precompute:
        precompute(
            args.candidates,
            args.features_dir or "data",
            jd_path=args.jd,
            train=args.train,
            force=args.force,
        )
    else:
        rank(
            args.candidates,
            args.out,
            args.features_dir or "data",
            jd_path=args.jd,
            model_path=args.model,
            output_json=args.json,
        )


if __name__ == "__main__":
    main()
