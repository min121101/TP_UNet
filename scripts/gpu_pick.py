#!/usr/bin/env python3
"""Query local NVIDIA GPUs via nvidia-smi and suggest the least busy device.

Run on the training host (not from a sandbox without GPU driver), e.g.:

  python scripts/gpu_pick.py
  eval "$(python scripts/gpu_pick.py --export-sh)"
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def _run_nvidia_smi() -> str:
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not (r.stdout or "").strip():
        raise RuntimeError(
            r.stderr.strip()
            or "nvidia-smi failed (no driver / no GPU?). Run this on the GPU node."
        )
    return r.stdout


def _parse_lines(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, used, total, util = parts[0], parts[1], parts[2], parts[3].replace("%", "").strip()
        rows.append(
            {
                "index": int(idx),
                "mem_used_mib": int(float(used)),
                "mem_total_mib": int(float(total)),
                "util": int(float(util)),
            }
        )
    if not rows:
        raise RuntimeError("No GPU rows parsed from nvidia-smi output.")
    return rows


def score(row: dict) -> float:
    # Prefer low used memory; lightly penalize high utilization.
    return row["mem_used_mib"] + row["util"] * 50.0


def select_best_index(rows: list[dict]) -> int:
    return min(rows, key=score)["index"]


def pick_best_gpu_string() -> str:
    """Return one physical GPU index as string (for CUDA_VISIBLE_DEVICES)."""
    raw = _run_nvidia_smi()
    rows = _parse_lines(raw)
    return str(select_best_index(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description="Pick least-loaded NVIDIA GPU by nvidia-smi.")
    ap.add_argument(
        "--export-sh",
        action="store_true",
        help="Print shell: export CUDA_VISIBLE_DEVICES=<idx>",
    )
    ap.add_argument(
        "--threshold-free-mib",
        type=int,
        default=4096,
        help="Report GPUs with at least this much free memory as 'likely free'.",
    )
    args = ap.parse_args()

    try:
        raw = _run_nvidia_smi()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    rows = _parse_lines(raw)
    best = select_best_index(rows)

    print("index | mem_used_MiB | mem_total_MiB | util_% | free_MiB | score")
    for r in sorted(rows, key=lambda x: x["index"]):
        free_m = r["mem_total_mib"] - r["mem_used_mib"]
        sc = score(r)
        mark = " <-- suggested" if r["index"] == best else ""
        print(
            f"  {r['index']:3d} | {r['mem_used_mib']:12d} | {r['mem_total_mib']:13d} | "
            f"{r['util']:6d} | {free_m:8d} | {sc:8.0f}{mark}"
        )

    thr = args.threshold_free_mib
    likely = [
        r["index"]
        for r in rows
        if (r["mem_total_mib"] - r["mem_used_mib"] >= thr and r["util"] < 95)
    ]
    print()
    print(f"GPU count: {len(rows)}")
    print(
        f"Likely usable (free>={thr} MiB and util<95%): "
        f"{likely if likely else '(none by this rule)'}"
    )
    print(f"Suggested CUDA_VISIBLE_DEVICES (single): {best}")

    if args.export_sh:
        print(f'export CUDA_VISIBLE_DEVICES="{best}"')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
