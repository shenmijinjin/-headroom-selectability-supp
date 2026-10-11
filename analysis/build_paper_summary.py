#!/usr/bin/env python3
"""Build paper-facing summaries from the restored raw prediction artifacts.

No new model fits are made. P3 raw per-row predictions and factorial raw cell
predictions have been restored (2026-10-11) and re-analyzed with the passage-
clustered bootstrap; points and CIs below come from those raw records.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from bootstrap_utils import TASKS, source_sha256


def legacy_p3(path):
    data = json.loads(path.read_text())
    if "aggregate" in data:
        return {"status": "raw-recomputed", "source": path.parent.name + "/" + path.name, "source_sha256": source_sha256(path),
                "aggregate": data["aggregate"]}
    inits = data["init_cells"]
    per_task = {task: float(np.mean([init["per_task"][task] for init in inits])) for task in TASKS}
    task_equal = float(np.mean(list(per_task.values())))
    return {"status": "points-derived-from-retained-scalar-summaries",
            "source": path.parent.name + "/" + path.name, "source_sha256": source_sha256(path), "n_fixed_inits": len(inits),
            "protocol": data["meta"], "per_task_no_gate": per_task,
            "per_init_task_equal_no_gate": {str(init["init"]): float(np.mean([init["per_task"][task] for task in TASKS])) for init in inits},
            "no_gate_task_equal": {"point": task_equal, "ci": None, "estimand": "task_equal_row",
                                   "recomputed_point": True, "recomputed_from_raw": False,
                                   "formula": "mean over four per-task row means, then over fixed inits"},
            "no_gate_row_pooled": {"point": float(np.mean([i["sel"] for i in inits])), "ci": None,
                                   "estimand": "row_pooled", "recomputed_point": True,
                                   "recomputed_from_raw": False},
            "gate_row_pooled": {"point": float(np.mean([i["gated"] for i in inits])), "ci": None,
                                "estimand": "row_pooled", "recomputed_point": True,
                                "recomputed_from_raw": False,
                                "task_equal_point": None,
                                "reason": "Per-task gated means and raw gate predictions not retained"},
            "gate_coverage_row_pooled": float(np.mean([i["coverage"] for i in inits])),
            "random_row_pooled": float(np.mean([i["random"] for i in inits])),
            "oracle_row_pooled": float(np.mean([i["oracle"] for i in inits])),
            "ci_status": "unavailable; no raw passage prediction records; legacy intervals omitted",
            "legacy_gate_objective": "row_pooled dev reward; repaired driver preserves this default (task_equal_row is an explicitly labelled optional objective)"}


def restored_p3(root, writer, suffix):
    """P3 entry from restored raw predictions: passage-clustered bootstrap, raw-recomputed."""
    name = "g4_passage_" + writer + suffix
    summary = json.loads((root / name / "bootstrap_summary.json").read_text())
    report = json.loads((root / name / "report.json").read_text())
    pred = root / name / "predictions.json"
    te = summary["task_equal_row"]["metrics"]
    rp = summary["row_pooled"]["metrics"]
    per_task = {task: {"selected": summary["per_task"][task]["selected"],
                       "gated": summary["per_task"][task]["gated"],
                       "coverage": summary["per_task"][task]["coverage"]}
                for task in summary["task_equal_row"]["task_order"]}
    return {"status": "raw-restored",
            "source": name + "/predictions.json", "source_sha256": source_sha256(pred),
            "n_fixed_inits": summary["n_fixed_inits"],
            "protocol": report["meta"],
            "n_rows": summary["task_equal_row"]["n_rows"],
            "n_passages": summary["task_equal_row"]["n_passages"],
            "bootstrap": {"unit": "passage", "replicates": summary["task_equal_row"]["bootstrap_replicates"],
                          "seed": summary["task_equal_row"]["bootstrap_seed"],
                          "init_aggregation": summary["init_aggregation"]},
            "no_gate_task_equal": {"point": te["selected"]["point"], "ci": te["selected"]["ci"],
                                   "estimand": "task_equal_row",
                                   "recomputed_point": True, "recomputed_from_raw": True},
            "gate_task_equal": {"point": te["gated"]["point"], "ci": te["gated"]["ci"],
                                "estimand": "task_equal_row",
                                "recomputed_point": True, "recomputed_from_raw": True},
            "gate_coverage_task_equal": {"point": te["coverage"]["point"], "ci": te["coverage"]["ci"]},
            "random_task_equal": {"point": te["random"]["point"], "ci": te["random"]["ci"]},
            "oracle_task_equal": {"point": te["oracle"]["point"], "ci": te["oracle"]["ci"]},
            "per_task": per_task,
            "row_pooled_reference": {
                "no_gate": {"point": rp["selected"]["point"], "ci": rp["selected"]["ci"]},
                "gate": {"point": rp["gated"]["point"], "ci": rp["gated"]["ci"]},
                "estimand": "row_pooled",
                "note": "secondary; dev gate threshold was tuned under the row_pooled objective"},
            "ci_status": "available; passage-clustered bootstrap over restored raw predictions",
            "legacy_gate_objective": "row_pooled dev reward; repaired driver preserves this default (task_equal_row is an explicitly labelled optional objective)"}


def restored_factorial(root, writer):
    """Factorial entry from restored raw cell predictions with shared passage resampling."""
    ci_path = root / "factorial_statistics" / (writer + "_factorial_ci.json")
    pred_path = root / "factorial_statistics" / (writer + "_predictions.json")
    data = json.loads(ci_path.read_text())
    te = data["task_equal_row"]
    return {"status": "raw-restored",
            "source": "factorial_statistics/" + pred_path.name, "source_sha256": source_sha256(pred_path),
            "n_fixed_inits": data["n_fixed_inits"],
            "primary_estimand": "task_equal_row",
            "strata": data["strata"],
            "n_rows": te["n_rows"], "n_passages": te["n_passages"],
            "bootstrap": {"unit": "passage", "replicates": te["bootstrap_replicates"],
                          "seed": te["bootstrap_seed"], "init_aggregation": data["init_aggregation"]},
            "cells": te["cells"], "contrasts": te["contrasts"],
            "row_pooled_reference": {"cells": data["row_pooled"]["cells"],
                                     "contrasts": data["row_pooled"]["contrasts"],
                                     "estimand": "row_pooled"},
            "ci_status": "available; shared passage-clustered bootstrap draws across paired cells"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.results_dir)
    result = {"schema_version": 2,
              "provenance": "Restored raw prediction artifacts re-analyzed locally; no GPU experiment or imputed predictions",
              "p3": {}, "q1": {}, "available_raw_statistics": {},
              "factorial": {},
              "bootstrap_order_note": "Utilities sort passage_id strings before seeded draws. Querycount/null use string indices (0,1,10,...), whereas earlier audit used stored row order; minor finite-bootstrap endpoint differences are expected."}
    for writer in ["qwen", "gemma"]:
        result["p3"][writer] = {}
        for arm, suffix in [("context", ""), ("context_free", "_ctxfree")]:
            result["p3"][writer][arm] = restored_p3(root, writer, suffix)
        result["factorial"][writer] = restored_factorial(root, writer)
        path = root / "q1_query_consistency" / (writer + ".json")
        q1 = json.loads(path.read_text())
        result["q1"][writer] = {"source": path.parent.name + "/" + path.name, "source_sha256": source_sha256(path),
                                 "recomputed_from_raw": True, "macro_summaries": q1["macro_summaries"],
                                 "canonical_single_split": q1["canonical_single_split"]}
    for family in ["query_count", "perm_null"]:
        path = root / family / "report.json"
        result["available_raw_statistics"][family] = {"source": path.parent.name + "/" + path.name, "source_sha256": source_sha256(path),
                                                       "recomputed_from_raw": True,
                                                       "bootstrap_unit": "passage", "replicate_unit": "fixed Monte Carlo draws"}
    path = root / "canon_p3" / "report.json"
    canon = json.loads(path.read_text())
    result["small_canonical_diagnostic"] = {
        "status": "points-derived-from-retained-scalar-summaries",
        "source": "canon_p3/report.json", "source_sha256": source_sha256(path),
        "protocol": canon["meta"], "estimand": "row_pooled_fixed_init_mean",
        "scope": "shared-row context split and action-index split; not full passage-disjoint P3",
        "ci": None,
        "arms": {arm: {metric: float(np.mean([i[metric] for i in canon[arm]]))
                        for metric in ("sel", "gated", "random", "oracle", "coverage")}
                 for arm in ("native", "canon")}}
    output = Path(args.output or str(root / "paper_summary.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("WROTE", output)
    for writer, arms in result["p3"].items():
        for arm, report in arms.items():
            print(writer, arm, "no-gate task-equal", report.get("no_gate_task_equal"),
                  "gate task-equal", report.get("gate_task_equal"))
    for writer, fac in result["factorial"].items():
        print(writer, "factorial cells sel-base",
              {c: fac["cells"][c]["selected_minus_base"]["point"] for c in fac["cells"]})


if __name__ == "__main__":
    main()
