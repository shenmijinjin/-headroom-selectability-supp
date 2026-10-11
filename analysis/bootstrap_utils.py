"""Fixed-fit, passage-clustered percentile bootstrap with explicit estimands.

Input rows may be query/action-seed records or Monte Carlo repetitions. Repeated
rows never become independent bootstrap units. All metric columns use the same
passage multiplicities, including aligned, already-averaged critic init outputs.
For task-equal estimands each task contributes exactly 1 / n_tasks. Strata, when
supplied, preserve disjoint train/test passage cohorts in factorial contrasts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ["squad", "drop", "ropes", "pwc"]
ESTIMANDS = {"row_pooled", "passage_equal", "task_equal_row", "task_equal_passage"}


def anonymous_passage_id(task, context):
    """Stable task-specific hash; no passage text is written to prediction files."""
    if isinstance(context, str):
        content = context
    else:
        content = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return task + "::" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clustered_summary(values, passage_ids, tasks=None, *, estimand="passage_equal",
                      strata=None, task_order=None, n_boot=10000, seed=0,
                      return_samples=False):
    """Return matched points and CIs; supports structural NaN cells in factorial data.

    ``task_equal_row``: mean over recorded rows within each task, then mean over
    tasks; unequal passage sizes retain their row weights in both point and CI.
    ``task_equal_passage``: mean rows within passage, then passages within task.
    ``row_pooled`` and ``passage_equal`` omit task balancing and are explicitly
    different estimands. Bootstrap is stratified by task for task-equal summaries,
    and by any additional supplied stratum, with shared draws across columns.
    Initializations are fixed fits, not extra independent data; callers align and
    average them before calling this function.
    """
    if estimand not in ESTIMANDS:
        raise ValueError("Unknown estimand: " + str(estimand))
    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    array = np.asarray(values, dtype=float)
    one_column = array.ndim == 1
    if one_column:
        array = array[:, None]
    if array.ndim != 2 or not len(array) or np.isinf(array).any():
        raise ValueError("Need a nonempty finite/structural-NaN row-by-metric matrix")
    pids = list(map(str, passage_ids))
    labels = list(map(str, tasks)) if tasks is not None else ["all"] * len(array)
    blocks = list(map(str, strata)) if strata is not None else ["all"] * len(array)
    if not (len(array) == len(pids) == len(labels) == len(blocks)):
        raise ValueError("Values, passages, tasks, and strata must have equal lengths")
    if task_order is None:
        task_order = sorted(set(labels))
    if set(task_order) != set(labels):
        raise ValueError("Missing/unexpected task; an incomplete macro is not valid")
    identity = {}
    for pid, task, block in zip(pids, labels, blocks):
        pair = (task, block)
        if pid in identity and identity[pid] != pair:
            raise ValueError("A passage cannot span tasks or split strata")
        identity[pid] = pair
    units = sorted(identity)
    unit_index = {pid: i for i, pid in enumerate(units)}
    inverse = np.array([unit_index[pid] for pid in pids])
    sums = np.zeros((len(units), array.shape[1]))
    counts = np.zeros_like(sums)
    for column in range(array.shape[1]):
        valid = np.isfinite(array[:, column])
        sums[:, column] = np.bincount(inverse[valid], weights=array[valid, column], minlength=len(units))
        counts[:, column] = np.bincount(inverse[valid], minlength=len(units))
    task_equal = estimand.startswith("task_equal")
    passage_equal = estimand in {"passage_equal", "task_equal_passage"}
    if passage_equal:
        sums = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        counts = (counts > 0).astype(float)
    unit_tasks = np.array([identity[pid][0] for pid in units])
    buckets = {}
    for index, pid in enumerate(units):
        task, block = identity[pid]
        key = (task if task_equal else "all", block)
        buckets.setdefault(key, []).append(index)

    def estimate(weight):
        group_points = []
        for task in task_order if task_equal else [None]:
            mask = unit_tasks == task if task is not None else np.ones(len(units), bool)
            # Explicit contraction avoids platform BLAS warnings and preserves
            # the same passage multiplicities for every metric column.
            numerator = np.einsum("bi,ij->bj", weight[:, mask], sums[mask])
            denominator = np.einsum("bi,ij->bj", weight[:, mask], counts[mask])
            if (denominator <= 0).any():
                raise ValueError("A bootstrap draw has no observations for a metric/task")
            group_points.append(numerator / denominator)
        return np.mean(group_points, axis=0)

    point = estimate(np.ones((1, len(units))))[0]
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, array.shape[1]))
    for offset in range(0, n_boot, 500):
        size = min(500, n_boot - offset)
        weights = np.zeros((size, len(units)))
        for indices in buckets.values():
            indices = np.asarray(indices)
            draws = rng.integers(0, len(indices), (size, len(indices)))
            multiplicities = np.zeros((size, len(indices)))
            np.add.at(multiplicities, (np.arange(size)[:, None], draws), 1)
            weights[:, indices] = multiplicities
        samples[offset:offset + size] = estimate(weights)
    if not np.isfinite(point).all() or not np.isfinite(samples).all():
        raise ValueError("Nonfinite point/bootstrap estimate")
    intervals = np.quantile(samples, [.025, .975], axis=0).T
    result = {"point": float(point[0]) if one_column else point.tolist(),
              "ci": intervals[0].tolist() if one_column else intervals.tolist(),
              "n_passages": len(units), "n_rows": len(array),
              "estimand": estimand, "bootstrap_unit": "passage",
              "bootstrap_replicates": n_boot, "bootstrap_seed": seed,
              "task_order": list(task_order),
              "n_passages_per_task": {t: int((unit_tasks == t).sum()) for t in task_order},
              "uncertainty_scope": "sampling passages conditional on fixed fits, action seeds and MC draws"}
    if return_samples:
        result["bootstrap_samples"] = samples[:, 0] if one_column else samples
    return result


def averaged_prediction_rows(per_init_rows, metrics):
    """Align prediction records across fixed critic inits, then average each row.

    Record identity = (task, passage_id, row, action_seed); missing or duplicated
    identities fail explicitly. Critic inits cannot enter the bootstrap as extra
    observations. This schema is emitted by the P3 and factorial training scripts.
    """
    if not per_init_rows:
        raise ValueError("No initialization-level raw prediction records")
    def key(record):
        return (record["task"], record["passage_id"], int(record["row"]), int(record["action_seed"]))
    maps = []
    for rows in per_init_rows:
        if any(not np.isfinite(row[metric]) for row in rows for metric in metrics):
            raise ValueError("Observed raw prediction metrics must be finite")
        mapping = {key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate raw prediction identity")
        maps.append(mapping)
    keys = sorted(maps[0])
    if any(set(mapping) != set(keys) for mapping in maps[1:]):
        raise ValueError("Initialization outputs do not cover the same prediction rows")
    values = np.array([[np.mean([mapping[k][metric] for mapping in maps])
                        for metric in metrics] for k in keys])
    return values, [k[1] for k in keys], [k[0] for k in keys], keys
