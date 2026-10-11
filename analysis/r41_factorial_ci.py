#!/usr/bin/env python3
"""Recompute factorial CIs only from aligned initialization-level raw predictions.

All cells and fixed critic fits share passage multiplicities in each replicate.
Task x seen/unseen passage cohorts are resampled separately; cells A/B share the
seen-passage draw and C/D the unseen draw. Fixed init predictions are averaged
before bootstrap. A report containing scalar cell means cannot identify a CI.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from bootstrap_utils import TASKS, averaged_prediction_rows, clustered_summary, source_sha256

CELLS = ["A", "B", "C", "D"]
CONTRASTS = [("A", "C"), ("B", "D"), ("A", "B"), ("C", "D")]


def load_raw(path):
    data = json.loads(Path(path).read_text())
    if not data.get("init_predictions"):
        raise ValueError("Raw init_predictions missing; scalar summaries cannot produce sampling CIs")
    for init in data["init_predictions"]:
        for row in init["rows"]:
            for field in ["task", "passage_id", "row", "action_seed", "cell", "selected", "random"]:
                if field not in row:
                    raise ValueError("Missing raw prediction field: " + field)
    return data


def summarize_factorial(data, n_boot=10000, seed=0):
    inits = data["init_predictions"]
    values, pids, tasks, keys = averaged_prediction_rows([i["rows"] for i in inits], ["selected", "random"])
    def identity(r):
        return (r["task"], r["passage_id"], int(r["row"]), int(r["action_seed"]))
    cell_maps = [{identity(r): r["cell"] for r in init["rows"]} for init in inits]
    if any(m != cell_maps[0] for m in cell_maps[1:]):
        raise ValueError("Cell assignments differ across critic initializations")
    cells = [cell_maps[0][key] for key in keys]
    if set(cells) != set(CELLS):
        raise ValueError("All four factorial cells must be present")
    matrix = np.full((len(keys), 8), np.nan)
    for i, cell in enumerate(cells):
        j = CELLS.index(cell)
        matrix[i, j] = values[i, 0]
        matrix[i, j + 4] = values[i, 0] - values[i, 1]
    cohorts = ["seen" if c in {"A", "B"} else "unseen" for c in cells]
    out = {"schema_version": 2, "n_fixed_inits": len(inits),
           "primary_estimand": "task_equal_row", "cells": CELLS,
           "init_aggregation": "align identities and average predictions before shared passage resampling",
           "strata": "task x seen/unseen passages; seen and unseen are disjoint cohorts",
           "uncertainty_scope": "passage sampling conditional on fixed fits and action seeds"}
    for estimand in ["task_equal_row", "row_pooled"]:
        result = clustered_summary(matrix, pids, tasks, estimand=estimand, strata=cohorts,
                                   task_order=TASKS, n_boot=n_boot, seed=seed, return_samples=True)
        samples = result.pop("bootstrap_samples")
        point, ci = result.pop("point"), result.pop("ci")
        result["cells"] = {cell: {"selected_minus_base": {"point": point[j], "ci": ci[j]},
                                    "selected_minus_random": {"point": point[j + 4], "ci": ci[j + 4]}}
                           for j, cell in enumerate(CELLS)}
        result["contrasts"] = {}
        for a, b in CONTRASTS:
            i, j = CELLS.index(a), CELLS.index(b)
            result["contrasts"][a + "-" + b] = {"point": point[i] - point[j],
                                                   "ci": np.quantile(samples[:, i] - samples[:, j], [.025,.975]).tolist()}
        out[estimand] = result
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="Single writer's predictions.json; must contain raw rows")
    parser.add_argument("--output")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    if args.raw:
        data = load_raw(args.raw)
        result = summarize_factorial(data, args.n_boot)
        result["source"] = {"logical_artifact": "factorial_predictions/" + Path(args.raw).name,
                            "sha256": source_sha256(args.raw)}
        output = Path(args.output or str(Path(args.raw).with_name("bootstrap_summary.json")))
    else:
        result = {}
        root = Path(os.environ.get("RESULTS_DIR", "results"))
        for writer in ["qwen", "gemma"]:
            path = root / ("e1_factorial_passage_" + writer) / "predictions.json"
            if path.is_file():
                result[writer] = summarize_factorial(load_raw(path), args.n_boot)
                result[writer]["source_sha256"] = source_sha256(path)
            else:
                result[writer] = {"status": "unavailable", "ci": None,
                                  "reason": "Raw passage/task/init prediction records are not retained locally; no CI is inferred from scalar cell summaries"}
        output = Path(args.output or str(root / "factorial_statistics" / "report.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("WROTE", output)


if __name__ == "__main__":
    main()
