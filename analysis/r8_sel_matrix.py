#!/usr/bin/env python3
"""Selector capacity control using the repaired Qwen passage-disjoint P3 driver.

Both arms share exact stored-context passage thirds, train/dev/test action seeds,
training budget, legacy row-pooled dev gate objective and matched task-equal passage CIs. Historical
row-split selector summaries cannot establish this repaired control; new GPU fits
are required. This wrapper never claims to reproduce those legacy row summaries.
"""
import argparse
import json
from pathlib import Path
import sys
import r41_g4_passage as p3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["current", "stronger"])
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out_dir", help="Per-arm output directory containing report and raw predictions")
    output.add_argument("--out", help="Compatibility report path; raw records are placed in <stem>_artifacts")
    parser.add_argument("--init_seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    cfg = {"current": {"d_hidden":256,"n_layers":2}, "stronger":{"d_hidden":512,"n_layers":4}}[args.arm]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out).with_name(Path(args.out).stem + "_artifacts")
    sys.argv = [sys.argv[0], "--out_dir", str(out_dir), "--d_hidden", str(cfg["d_hidden"]),
                "--n_layers", str(cfg["n_layers"]), "--init_seeds", args.init_seeds, "--epochs", str(args.epochs)]
    p3.main()
    report = json.loads((out_dir / "report.json").read_text())
    report["arm"] = args.arm
    report["cfg"] = cfg
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.out:
        report["meta"]["init_predictions_file"] = out_dir.name + "/predictions.json"
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
