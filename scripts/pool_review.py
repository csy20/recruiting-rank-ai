#!/usr/bin/env python3
"""Generate an HTML review report from a ranked submission CSV."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime


def load_candidates(candidates_path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with open(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid:
                    result[str(cid)] = c
            except json.JSONDecodeError:
                continue
    return result


def load_ranking(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _dimension_bar(dim_name: str, score: float) -> str:
    pct = int(score * 100)
    color = "#22c55e" if score > 0.6 else "#eab308" if score > 0.3 else "#ef4444"
    return (
        f'<div class="dim-row">'
        f'<span class="dim-label">{dim_name}</span>'
        f'<div class="dim-bar-bg"><div class="dim-bar" '
        f'style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="dim-val">{pct}%</span>'
        f"</div>"
    )


def _signal_bar(name: str, score: float) -> str:
    pct = int(score * 100)
    color = "#22c55e" if score > 0.6 else "#eab308" if score > 0.3 else "#ef4444"
    return (
        f'<div class="sig-row">'
        f'<span class="sig-label">{name}</span>'
        f'<div class="sig-bar-bg"><div class="sig-bar" '
        f'style="width:{pct}%;background:{color}"></div></div>'
        f"</div>"
    )


def _score_distribution(scores: list[float]) -> str:
    if not scores:
        return "<p>No scores</p>"
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts = [0] * (len(bins) - 1)
    for s in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break
    max_count = max(counts) or 1
    bars = []
    for i in range(len(bins) - 1):
        pct = int(counts[i] / max_count * 100)
        bars.append(
            f'<div class="hist-bar-wrapper">'
            f'<span class="hist-label">{bins[i]:.1f}-{bins[i + 1]:.1f}</span>'
            f'<div class="hist-bar-bg"><div class="hist-bar" style="height:{pct}px"></div></div>'
            f'<span class="hist-count">{counts[i]}</span>'
            f"</div>"
        )
    return '<div class="histogram">' + "".join(bars) + "</div>"


def generate_report(
    ranking: list[dict],
    candidates: dict[str, dict] | None = None,
    title: str = "Pool Review Report",
) -> str:
    scores = [float(r.get("score", 0)) for r in ranking]

    dim_keys = [
        "technical_match",
        "semantic_match",
        "career_quality",
        "behavioral",
        "retention",
    ]

    tbody_rows = ""
    for i, r in enumerate(ranking):
        cid = r.get("candidate_id", "")
        rank = r.get("rank", i + 1)
        score = float(r.get("score", 0))
        reasoning = r.get("reasoning", "")

        detail_rows = ""
        dim_cells = '<td class="dim-td">'
        if candidates and cid in candidates:
            cand = candidates[cid]
            profile = cand.get("profile", {}) or {}
            signals = cand.get("redrob_signals", {}) or {}
            current_title = profile.get("current_title", "") or ""
            current_company = profile.get("current_company", "") or ""
            summary = (profile.get("summary", "") or "")[:200]

            detail_rows += (
                f'<tr class="detail-row" id="detail-{i}">'
                f'<td colspan="4">'
                f'<div class="detail-grid">'
                f'<div class="detail-col"><strong>Title:</strong> {current_title}<br>'
                f"<strong>Company:</strong> {current_company}<br>"
                f"<strong>Summary:</strong> {summary}</div>"
                f'<div class="detail-col">'
            )
            for dk in dim_keys:
                dv = cand.get("_dimensions", {}).get(dk, 0.0)
                detail_rows += _dimension_bar(dk, dv)
            detail_rows += "</div><div class='detail-col'>"
            for sk in [
                "recruiter_response_rate",
                "interview_completion_rate",
                "open_to_work_flag",
                "saved_by_recruiters_30d",
                "github_activity_score",
            ]:
                sv = signals.get(sk, 0)
                sn = sk.replace("_", " ").title()
                if isinstance(sv, bool):
                    sv = 1.0 if sv else 0.0
                sv_f = float(sv) if sv is not None else 0.0
                detail_rows += _signal_bar(sn, min(sv_f, 1.0))
            detail_rows += "</div></div></td></tr>"

        dim_cells += "</td>"

        row_class = ""
        if score < 0.3:
            row_class = ' class="row-low"'
        elif score > 0.7:
            row_class = ' class="row-high"'

        tbody_rows += (
            f'<tr{row_class} onclick="toggle({i})">'
            f"<td>{rank}</td>"
            f"<td>{cid}</td>"
            f"<td>{score:.4f}</td>"
            f'<td class="reason-cell">{reasoning[:120]}...</td>'
            f"</tr>"
            f"{detail_rows}"
        )

    dist_html = _score_distribution(scores)

    css = (
        "*{margin:0;padding:0;box-sizing:border-box;}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "background:#f8fafc;color:#1e293b;padding:20px;}"
        "h1{font-size:24px;margin-bottom:4px;}"
        ".subtitle{color:#64748b;font-size:14px;margin-bottom:20px;}"
        ".stats{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;}"
        ".stat-card{background:white;border-radius:8px;padding:16px 24px;"
        "box-shadow:0 1px 3px #0000001a;flex:1;min-width:120px;}"
        ".stat-card .num{font-size:28px;font-weight:700;}"
        ".stat-card .label{font-size:12px;color:#64748b;text-transform:uppercase;}"
        "table{width:100%;border-collapse:collapse;background:white;border-radius:8px;"
        "overflow:hidden;box-shadow:0 1px 3px #0000001a;}"
        "th{background:#f1f5f9;padding:10px 12px;text-align:left;font-size:12px;"
        "text-transform:uppercase;color:#64748b;cursor:pointer;}"
        "td{padding:10px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;}"
        ".row-low{background:#fef2f2;}"
        ".row-high{background:#f0fdf4;}"
        ".reason-cell{color:#64748b;max-width:300px;overflow:hidden;"
        "text-overflow:ellipsis;white-space:nowrap;}"
        ".detail-row{display:none;}"
        ".detail-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:8px 0;}"
        ".detail-col{font-size:12px;line-height:1.6;}"
        ".dim-row,.sig-row{display:flex;align-items:center;gap:8px;margin:3px 0;}"
        ".dim-label,.sig-label{width:120px;font-size:11px;color:#475569;text-align:right;}"
        ".dim-bar-bg,.sig-bar-bg{flex:1;height:8px;background:#e2e8f0;border-radius:4px;}"
        ".dim-bar,.sig-bar{height:8px;border-radius:4px;}"
        ".dim-val{width:30px;font-size:11px;color:#64748b;}"
        ".histogram{display:flex;align-items:flex-end;gap:4px;height:120px;padding:16px 0;}"
        ".hist-bar-wrapper{flex:1;display:flex;flex-direction:column;align-items:center;}"
        ".hist-label{font-size:9px;color:#64748b;}"
        ".hist-bar-bg{width:100%;max-width:40px;height:100px;background:#f1f5f9;"
        "border-radius:4px 4px 0 0;display:flex;align-items:flex-end;}"
        ".hist-bar{width:100%;background:#3b82f6;border-radius:4px 4px 0 0;min-height:1px;}"
        ".hist-count{font-size:10px;color:#64748b;}"
        "tr:not(.detail-row):hover{background:#f8fafc;cursor:pointer;}"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &middot; {len(ranking)} candidates</p>

<div class="stats">
  <div class="stat-card"><div class="num">{scores[0]:.4f}</div><div class="label">Top Score</div></div>
  <div class="stat-card"><div class="num">{scores[-1]:.4f}</div><div class="label">Bottom Score</div></div>
  <div class="stat-card"><div class="num">{sum(scores) / len(scores):.4f}</div><div class="label">Mean Score</div></div>
  <div class="stat-card"><div class="num">{len(ranking)}</div><div class="label">Total Ranked</div></div>
</div>

<h2>Score Distribution</h2>
{dist_html}

<h2>Ranked Candidates</h2>
<table>
<thead><tr><th>Rank</th><th>ID</th><th>Score</th><th>Reasoning</th></tr></thead>
<tbody>
{tbody_rows}
</tbody>
</table>

<script>
function toggle(idx) {{
  var el = document.getElementById('detail-' + idx);
  if (el) el.style.display = el.style.display === 'table-row' ? 'none' : 'table-row';
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML pool review report from a ranked submission CSV"
    )
    parser.add_argument("csv_path", help="Path to ranked CSV (output of rank.py)")
    parser.add_argument("--candidates", help="Path to candidates.jsonl for enriched review")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    parser.add_argument("--title", default="Pool Review Report", help="Report title")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV not found: {args.csv_path}")
        sys.exit(1)

    ranking = load_ranking(args.csv_path)
    if not ranking:
        print("ERROR: No rows in CSV")
        sys.exit(1)

    candidates = None
    if args.candidates:
        if not os.path.exists(args.candidates):
            print(f"ERROR: Candidates file not found: {args.candidates}")
            sys.exit(1)
        candidates = load_candidates(args.candidates)
        print(f"Loaded {len(candidates)} candidates from {args.candidates}")

    html = generate_report(ranking, candidates, title=args.title)

    out_path = args.output or args.csv_path.replace(".csv", "_review.html")
    if out_path == args.csv_path:
        out_path += "_review.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Report written to {out_path}")
    print(f"  Top score:  {ranking[0].get('score', 'N/A')}")
    print(f"  Bottom score: {ranking[-1].get('score', 'N/A')}")
    print(f"  Total: {len(ranking)} candidates")


if __name__ == "__main__":
    main()
