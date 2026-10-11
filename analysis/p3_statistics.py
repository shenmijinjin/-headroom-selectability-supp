"""Raw prediction schema and matched fixed-fit P3 summaries (CPU only).

Training scripts emit one record per (task, passage, QA row, action seed, init).
The primary estimand is task-equal, retaining QA-row/action-seed weights within
each task. Raw init predictions are averaged before a shared passage resample.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from bootstrap_utils import TASKS, averaged_prediction_rows, clustered_summary, source_sha256

METRICS = ["selected", "gated", "random", "oracle", "coverage"]


def validated_pool(path, rank=8):
    """Read a required reward/action pair; incomplete pools must not run silently."""
    path = Path(path)
    data = json.loads(path.read_text())  # Missing required files raise explicitly.
    actions = np.load(path.with_suffix(".actions.npz"))["actions"]
    rows = data["rows"]
    if not rows or len(rows) != len(actions):
        raise ValueError("Nonempty action/reward row counts must match: " + path.name)
    if actions.ndim != 3 or actions.shape[2] != rank or not np.isfinite(actions).all():
        raise ValueError("Expected finite [row,candidate,rank] actions: " + path.name)
    for row, action in zip(rows, actions):
        reward = np.asarray(row["candidate_rewards"], float)
        if reward.ndim != 1 or len(reward) != len(action):
            raise ValueError("Candidate reward/action counts must match: " + path.name)
        if not np.isfinite(reward).all() or not np.isfinite(row["baseline_reward"]):
            raise ValueError("Reward fields must be finite: " + path.name)
        if "context" not in row or "row" not in row:
            raise ValueError("Stored context and row identity are required: " + path.name)
    return data, actions


def task_equal_point(values, tasks):
    values = np.asarray(values, float)
    if set(tasks) != set(TASKS):
        raise ValueError("All four tasks must be present for a task-equal macro")
    return float(np.mean([values[np.array(tasks) == task].mean() for task in TASKS]))


def prediction_rows(entries, selected, gated, rewards, gate_scores, mask, choices=None):
    """Export sufficient predictions without exporting raw context strings."""
    rows = [dict(task=e["task"], passage_id=e["passage_id"], row=int(e["row"]),
                 action_seed=int(e["seed"]), selected=float(selected[i]),
                 gated=float(gated[i]), random=float(rewards[i].mean()),
                 oracle=float(rewards[i].max()), coverage=float(mask[i]),
                 gate_score=float(gate_scores[i])) for i, e in enumerate(entries)]
    for i, (entry, row) in enumerate(zip(entries, rows)):
        if choices is not None:
            row["selected_action_index"] = int(choices[i])
        if "baseline_reward" in entry:
            row["baseline_reward"] = float(entry["baseline_reward"])
    return rows


def fixed_init_points(rows):
    """Descriptive points only; aggregate uncertainty is computed after alignment."""
    tasks = [r["task"] for r in rows]
    return {"task_equal_row": {metric: task_equal_point([r[metric] for r in rows], tasks)
                               for metric in METRICS},
            "row_pooled": {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS},
            "per_task": {task: {metric: float(np.mean([r[metric] for r in rows if r["task"] == task]))
                                for metric in METRICS} for task in TASKS}}


def summarize_inits(init_predictions, n_boot=10000, seed=0):
    values, pids, tasks, _ = averaged_prediction_rows(
        [init["rows"] for init in init_predictions], METRICS)
    out = {"n_fixed_inits": len(init_predictions), "metrics": METRICS,
           "init_aggregation": "align identities and average predictions before resampling"}
    for estimand in ["task_equal_row", "row_pooled"]:
        result = clustered_summary(values, pids, tasks, estimand=estimand,
                                   task_order=TASKS, n_boot=n_boot, seed=seed)
        result["metrics"] = {metric: {"point": result["point"][i], "ci": result["ci"][i]}
                             for i, metric in enumerate(METRICS)}
        del result["point"], result["ci"]
        out[estimand] = result
    out["per_task"] = {}
    for task in TASKS:
        mask = np.array(tasks) == task
        result = clustered_summary(values[mask], np.array(pids)[mask], estimand="row_pooled",
                                   n_boot=n_boot, seed=seed)
        out["per_task"][task] = {metric: {"point": result["point"][i], "ci": result["ci"][i]}
                                 for i, metric in enumerate(METRICS)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="predictions.json with aligned fixed-init rows")
    parser.add_argument("--output")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    data = json.loads(Path(args.raw).read_text())
    if not data.get("init_predictions"):
        raise ValueError("Raw init_predictions missing; no CI can be recovered from scalar means")
    result = summarize_inits(data["init_predictions"], args.n_boot)
    result["schema_version"] = 2
    result["source"] = {"logical_artifact": "p3_predictions/" + Path(args.raw).name,
                        "sha256": source_sha256(args.raw)}
    output = Path(args.output or str(Path(args.raw).with_name("bootstrap_summary.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("WROTE", output)


if __name__ == "__main__":
    main()
