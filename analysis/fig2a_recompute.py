#!/usr/bin/env python3
"""Strict consistency check for the query-consistency figure artifact.

The historical filename is misleading: this checks q1 (query consistency), not
factorial cells. Select-half oracle differs from the per-query oracle. C is the
finite-pair mean Spearman correlation. All fixed partition seeds, canonical
split, point estimands and passage CIs use r8_q1_consistency's recorded protocol.
"""
import argparse
import json
import os
from pathlib import Path
from r8_q1_consistency import analyse_writer


def differences(actual, expected, path="", tolerance=1e-12):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return [dict(path=path, reason="field sets differ")]
        return [diff for key in actual for diff in differences(actual[key], expected[key], path + "/" + key, tolerance)]
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [dict(path=path, reason="lengths differ")]
        return [diff for i, (a, b) in enumerate(zip(actual, expected)) for diff in differences(a,b,path + "/" + str(i),tolerance)]
    if isinstance(actual, (int,float)) and isinstance(expected,(int,float)):
        if abs(actual - expected) <= tolerance:
            return []
    elif actual == expected:
        return []
    return [dict(path=path, actual=actual, expected=expected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.results_dir)
    report = {}
    for writer in ["qwen", "gemma"]:
        frozen = json.loads((root / "q1_query_consistency" / (writer + ".json")).read_text())
        computed = analyse_writer(writer, root)
        mismatch = differences(computed, frozen)
        report[writer] = {"status": "pass" if not mismatch else "mismatch", "tolerance": 1e-12,
                          "differences": mismatch}
        print(writer, report[writer]["status"], "mismatches", len(mismatch))
    output = Path(args.output or str(root / "q1_statistics_check" / "report.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(r["differences"] for r in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
