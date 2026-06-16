#!/usr/bin/env python3
"""Summarize residual-policy trace CSV files for local tuning."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


CONE_X = 2.23
CONE_Y = 1.17
CONE_VISUAL_HALF_SIDE = 0.15


def _float(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def summarize(path: Path, minor_threshold: float = 0.04, major_threshold: float = 0.08):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {"path": str(path), "rows": 0}

    final = rows[-1]
    penalty_rows = []
    minor_rows = []
    major_rows = []
    cone_visual_rows = []
    segment_counts = defaultdict(lambda: {"minor": 0, "major": 0, "penalty": 0.0, "max_dev": 0.0})

    for row in rows:
        x = _float(row, "x")
        y = _float(row, "y")
        deviation = _float(row, "lane_deviation")
        penalty = _float(row, "penalty_increment")
        segment = row.get("segment_name") or ""
        if deviation > minor_threshold:
            minor_rows.append(row)
            segment_counts[segment]["minor"] += 1
            segment_counts[segment]["max_dev"] = max(segment_counts[segment]["max_dev"], deviation)
        if deviation > major_threshold:
            major_rows.append(row)
            segment_counts[segment]["major"] += 1
        if penalty:
            penalty_rows.append(row)
            segment_counts[segment]["penalty"] += penalty
        if abs(x - CONE_X) < CONE_VISUAL_HALF_SIDE and abs(y - CONE_Y) < CONE_VISUAL_HALF_SIDE:
            cone_visual_rows.append(row)

    return {
        "path": str(path),
        "rows": len(rows),
        "raw_time": _float(final, "raw_time", None),
        "penalty_time": _float(final, "penalty_time", None),
        "final_time": _float(final, "final_time", None),
        "minor_count": len(minor_rows),
        "major_count": len(major_rows),
        "cone_visual_count": len(cone_visual_rows),
        "segments": dict(segment_counts),
        "penalty_rows": penalty_rows,
    }


def print_summary(summary, show_penalties: bool):
    print(f"\n{summary['path']}")
    print(
        "  final: "
        f"raw={summary.get('raw_time')} "
        f"penalty={summary.get('penalty_time')} "
        f"final={summary.get('final_time')}"
    )
    print(
        "  counts: "
        f">0.04={summary.get('minor_count')} "
        f">0.08={summary.get('major_count')} "
        f"cone_visual_box={summary.get('cone_visual_count')}"
    )
    for segment, data in sorted(
        summary.get("segments", {}).items(),
        key=lambda item: (-item[1]["penalty"], -item[1]["minor"], item[0]),
    ):
        print(
            f"  {segment}: "
            f"penalty={data['penalty']:.1f} "
            f">0.04={data['minor']} "
            f">0.08={data['major']} "
            f"max_dev={data['max_dev']:.3f}"
        )

    if show_penalties:
        for row in summary.get("penalty_rows", []):
            print(
                "    penalty "
                f"step={row.get('step')} "
                f"seg={row.get('segment_name')} "
                f"x={_float(row, 'x'):.3f} "
                f"y={_float(row, 'y'):.3f} "
                f"dev={_float(row, 'lane_deviation'):.3f} "
                f"inc={_float(row, 'penalty_increment'):.1f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--show-penalties", action="store_true")
    args = parser.parse_args()

    for trace in args.traces:
        print_summary(summarize(Path(trace)), args.show_penalties)


if __name__ == "__main__":
    main()
