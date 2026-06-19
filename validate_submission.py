#!/usr/bin/env python3
"""Validate submission CSV against the Redrob hackathon spec."""

import argparse
import csv
import json
import sys


def load_candidate_ids(candidates_path: str) -> set[str]:
    ids: set[str] = set()
    if candidates_path.endswith(".gz"):
        import gzip

        opener = gzip.open
    else:
        opener = open
    with opener(candidates_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid:
                    ids.add(str(cid))
            except json.JSONDecodeError:
                continue
    return ids


def validate(
    csv_path: str,
    candidates_path: str | None = None,
) -> int:
    errors: list[str] = []

    # Load candidate IDs if provided
    known_ids: set[str] | None = None
    if candidates_path:
        known_ids = load_candidate_ids(candidates_path)
        print(f"Loaded {len(known_ids)} candidate IDs from {candidates_path}")

    # Read CSV
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
            errors.append(
                f"Columns: expected ['candidate_id', 'rank', 'score', 'reasoning'], "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(row)

    # Exactly 100 data rows
    if len(rows) != 100:
        errors.append(f"Expected 100 data rows, got {len(rows)}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    # Validate each row
    seen_ranks: set[int] = set()
    seen_ids: set[str] = set()
    prev_score: float | None = None

    for i, row in enumerate(rows):
        row_num = i + 1

        # candidate_id
        cid = row.get("candidate_id", "").strip()
        if not cid:
            errors.append(f"Row {row_num}: empty candidate_id")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)
            if known_ids is not None and cid not in known_ids:
                errors.append(f"Row {row_num}: candidate_id '{cid}' not found in candidates.jsonl")

        # rank
        rank_str = row.get("rank", "").strip()
        try:
            rank = int(rank_str)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: rank '{rank_str}' is not a valid integer")
            rank = None
        if rank is not None:
            if rank < 1 or rank > 100:
                errors.append(f"Row {row_num}: rank {rank} out of range [1, 100]")
            if rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            seen_ranks.add(rank)

        # score
        score_str = row.get("score", "").strip()
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: score '{score_str}' is not a valid float")
            score = None
        if score is not None:
            if score < 0.0 or score > 1.0:
                errors.append(f"Row {row_num}: score {score} out of range [0, 1]")
            if prev_score is not None and score > prev_score + 1e-9:
                errors.append(
                    f"Row {row_num}: score {score} > previous score {prev_score} "
                    f"(must be non-increasing)"
                )
            prev_score = score

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print(f"OK: {len(rows)} rows, all valid (ranks 1-100, scores non-increasing, IDs unique)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate hackathon submission CSV against spec")
    parser.add_argument("csv_path", help="Path to submission CSV")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl (or .jsonl.gz) for ID validation",
    )
    args = parser.parse_args()
    sys.exit(validate(args.csv_path, args.candidates))


if __name__ == "__main__":
    main()
