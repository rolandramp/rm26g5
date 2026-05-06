"""Aggregate RAG sweep results into a single CSV for the poster.

Walks ``results/`` and emits one row per ``results.json``. Configuration
(embedding / generator / chunking) is read from the sibling
``experiment_info.txt`` for OFAT runs and parsed from the folder name
``embed_<E>__gen_<G>__chunk_<C>`` for the factorial sweep.

Run from repo root::

    python poster/aggregate_results.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_CSV = Path(__file__).resolve().parent / "results_summary.csv"

METRIC_FAMILIES = [
    "context_precision",
    "faithfulness",
    "semantic_similarity",
    "answer_correctness",
    "bleu_score",
    "rouge_score",
    "lexical_similarity",
]
STAT_SUFFIXES = ("min", "max", "avg", "var")

FOLDER_CONFIG_RE = re.compile(
    r"^embed_(?P<embedding>.+?)__gen_(?P<generator>.+?)__chunk_(?P<chunking>.+)$"
)
INFO_LINE_RE = re.compile(r"-\s*(?P<key>[^:]+):\s*(?P<value>.+)$")


def parse_experiment_info(info_path: Path) -> dict[str, str]:
    """Read a single-run ``experiment_info.txt`` into a flat dict."""
    fields: dict[str, str] = {}
    for raw in info_path.read_text(encoding="utf-8").splitlines():
        match = INFO_LINE_RE.search(raw.strip())
        if match:
            fields[match.group("key").strip()] = match.group("value").strip()
    return {
        "embedding": fields.get("Embedding Profile", ""),
        "generator": fields.get("Generator Profile", ""),
        "chunking": fields.get("Chunking Profile", ""),
    }


def parse_folder_config(folder_name: str) -> dict[str, str] | None:
    match = FOLDER_CONFIG_RE.match(folder_name)
    if not match:
        return None
    return match.groupdict()


def classify_sweep(rel_path: Path) -> tuple[str, str]:
    """Return ``(sweep, level)`` derived from the path under ``results/``."""
    parts = rel_path.parts
    if not parts:
        return ("unknown", "")
    head = parts[0]
    if head in {"chunking-sweep", "embedding-sweep", "llm-sweep"}:
        sweep = head.removesuffix("-sweep")
        level = parts[1] if len(parts) > 1 else ""
        return (sweep, level)
    # Factorial run lives under a top-level timestamped directory.
    if len(parts) >= 2 and FOLDER_CONFIG_RE.match(parts[1]):
        return ("factorial", parts[1])
    return ("other", head)


def flatten_metrics(data: dict) -> dict[str, float | int | str]:
    flat: dict[str, float | int | str] = {}
    for family in METRIC_FAMILIES:
        for stat in STAT_SUFFIXES:
            key = f"{family}_{stat}"
            flat[key] = data.get(key, "")
    flat["avg_latency"] = data.get("avg_latency", "")
    rq = data.get("retrieval_quality") or {}
    flat["avg_contexts_retrieved"] = rq.get("avg_contexts_retrieved", "")
    flat["avg_context_length"] = rq.get("avg_context_length", "")
    flat["questions_with_contexts"] = rq.get("questions_with_contexts", "")
    return flat


EXCLUDED_TOP_DIRS = {"2026-04-26_19-53-01"}


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for results_path in sorted(RESULTS_DIR.rglob("results.json")):
        run_dir = results_path.parent
        rel = run_dir.relative_to(RESULTS_DIR)
        if rel.parts and rel.parts[0] in EXCLUDED_TOP_DIRS:
            continue
        sweep, level = classify_sweep(rel)

        info_path = run_dir / "experiment_info.txt"
        config = (
            parse_experiment_info(info_path)
            if info_path.exists()
            else parse_folder_config(run_dir.name) or {}
        )

        # Timestamp is whichever path component matches a YYYY-MM-DD_HH-MM-SS pattern.
        timestamp = next(
            (p for p in rel.parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", p)),
            "",
        )

        with results_path.open(encoding="utf-8") as fh:
            metrics = json.load(fh)

        row: dict[str, object] = {
            "sweep": sweep,
            "level": level,
            "embedding": config.get("embedding", ""),
            "generator": config.get("generator", ""),
            "chunking": config.get("chunking", ""),
            "timestamp": timestamp,
            "run_path": str(rel).replace("\\", "/"),
        }
        row.update(flatten_metrics(metrics))
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise SystemExit(f"No results.json files found under {RESULTS_DIR}")

    fieldnames = [
        "sweep",
        "level",
        "embedding",
        "generator",
        "chunking",
        "timestamp",
        "run_path",
    ]
    fieldnames += [f"{f}_{s}" for f in METRIC_FAMILIES for s in STAT_SUFFIXES]
    fieldnames += [
        "avg_latency",
        "avg_contexts_retrieved",
        "avg_context_length",
        "questions_with_contexts",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = collect_rows()
    rows.sort(key=lambda r: (r["sweep"], r["level"], r["embedding"], r["chunking"]))
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()