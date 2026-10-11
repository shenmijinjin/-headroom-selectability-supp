#!/usr/bin/env python3
"""Source-backed query-access summaries with explicit partition and sampling uncertainty.
Canonical split: one seed-0 generator per task, consumed in stored passage order.
Twenty re-splits: historical seed-(1000+r) generator reset for each passage.
Both task-equal and passage-pooled summaries are reported. MC repetitions are
averaged within passage before passage bootstrap; partition SD is not a CI.
"""
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from bootstrap_utils import TASKS, clustered_summary, source_sha256
R_SPLITS = 20


def split_response(response, rng):
    perm = rng.permutation(response.shape[1]); nsel = (response.shape[1] + 1) // 2
    sel, ev = perm[:nsel], perm[nsel:]
    score = response[:, sel].mean(axis=1); action = int(score.argmax())
    return float(score[action]), float(response[action, ev].mean())


def safe_correlation(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); value = float(spearmanr(left[valid], right[valid]).statistic)
    return value if np.isfinite(value) else None


def fixed_passage_stats(response):
    rhos, top1, opposite, nonzero, slots = [], [], 0, 0, 0
    for i in range(response.shape[1]):
        for j in range(i + 1, response.shape[1]):
            ri, rj = response[:, i], response[:, j]
            rho = safe_correlation(ri, rj)
            if rho is not None: rhos.append(rho)
            top1.append(int(ri.argmax() == rj.argmax()))
            opposite += int(((ri * rj) < 0).sum())
            nonzero += int(((np.abs(ri) > 1e-9) & (np.abs(rj) > 1e-9)).sum())
            slots += len(ri)
    return dict(ctx_oracle=float(response.mean(axis=1).max()),
                per_query_oracle=float(response.max(axis=0).mean()),
                C=float(np.mean(rhos)) if rhos else None, top1=float(np.mean(top1)),
                signconf=float(opposite / slots),
                signconf_nonzero=float(opposite / nonzero) if nonzero else None)


def finite_ci(values):
    valid = [(i, v) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    return clustered_summary([v for _, v in valid], [str(i) for i, _ in valid])['ci'] if len(valid) >= 3 else None


def split_summary(same, held):
    same, held = np.asarray(same), np.asarray(held)
    return dict(same=float(same.mean()), held=float(held.mean()),
                same_sd=float(same.std(ddof=0)), held_sd=float(held.std(ddof=0)),
                positive_fraction=float((held > 0).mean()),
                retention_ratio_of_means=float(held.mean() / same.mean()) if abs(same.mean()) > 1e-9 else None,
                same_by_split=same.tolist(), held_by_split=held.tolist(),
                sd_definition='population SD across query partitions on shared passages; not a sampling CI')


def analyse_writer(writer, results_dir):
    base = Path(results_dir) / 'e2_perquery'
    out = dict(writer=writer, n_splits=R_SPLITS, tasks={},
               meta=dict(schema_version=2, bootstrap_unit='passage', bootstrap_replicates=10000,
                         bootstrap_seed=0, canonical_seed=0,
                         canonical_rng_scope='one generator per task, consumed in stored passage order',
                         repeated_split_seeds=list(range(1000, 1000 + R_SPLITS)),
                         repeated_rng_scope='fresh generator per passage and split; historical convention',
                         signconf_definition='opposite signs over all candidate/query-pair slots; zeros are not sign conflicts',
                         source_sha256={}))
    all_same, all_held, all_oracles, all_pids, all_tasks, canonical_rows = [], [], [], [], [], []
    task_same, task_held = [], []
    for task in TASKS:
        source = base / f'{writer}_{task}_s42.json'
        out['meta']['source_sha256'][task] = source_sha256(source)
        contexts = [c for c in json.loads(source.read_text())['contexts'] if c['n_queries'] >= 2]
        canonical_rng = np.random.default_rng(0)
        fixed, same, held, canonical, records = [], [], [], [], []
        for context in contexts:
            cr, br = np.asarray(context['candidate_rewards'], float), np.asarray(context['baseline_rewards'], float)
            if cr.shape[1] != context['n_queries'] or len(br) != cr.shape[1]: raise ValueError('Inconsistent query dimensions')
            response = cr - br[None, :]; st = fixed_passage_stats(response); fixed.append(st)
            draws = [split_response(response, np.random.default_rng(1000 + r)) for r in range(R_SPLITS)]
            same.append([x[0] for x in draws]); held.append([x[1] for x in draws])
            canonical.append(split_response(response, canonical_rng))
            pid = f'{task}::{context["ctx_index"]}'
            records.append(dict(passage_id=pid, ctx_index=context['ctx_index'], first_row=context['first_row'],
                                n_queries=context['n_queries'], **st, same_by_split=same[-1], held_by_split=held[-1],
                                canonical_same=canonical[-1][0], canonical_held=canonical[-1][1]))
            all_pids.append(pid); all_tasks.append(task)
        same, held, canonical = np.asarray(same), np.asarray(held), np.asarray(canonical)
        oracle = np.array([x['ctx_oracle'] for x in fixed]); pids = [r['passage_id'] for r in records]
        repeated = split_summary(same.mean(axis=0), held.mean(axis=0))
        sampled = clustered_summary(np.column_stack([same.mean(axis=1), held.mean(axis=1), oracle]), pids)
        single = clustered_summary(canonical, pids)
        finite_C = [x['C'] for x in fixed if x['C'] is not None]
        nonzero_conf = [x['signconf_nonzero'] for x in fixed if x['signconf_nonzero'] is not None]
        out['tasks'][task] = dict(n_ctx=len(contexts), **repeated, retention=repeated['retention_ratio_of_means'],
            same_passage_ci=sampled['ci'][0], held_passage_ci=sampled['ci'][1],
            ctx_oracle=float(oracle.mean()), ctx_oracle_ci=sampled['ci'][2],
            per_query_oracle=float(np.mean([x['per_query_oracle'] for x in fixed])),
            C=float(np.mean(finite_C)) if finite_C else None, C_ci=finite_ci([x['C'] for x in fixed]),
            n_finite_C=len(finite_C), top1=float(np.mean([x['top1'] for x in fixed])),
            top1_definition='deterministic argmax equality across query pairs; not tie-aware',
            signconf=float(np.mean([x['signconf'] for x in fixed])),
            signconf_nonzero=float(np.mean(nonzero_conf)) if nonzero_conf else None,
            canonical_single_split=dict(seed=0, same=single['point'][0], held=single['point'][1],
                                        same_ci=single['ci'][0], held_ci=single['ci'][1]), per_passage=records)
        task_same.append(same.mean(axis=0)); task_held.append(held.mean(axis=0))
        all_same.extend(same); all_held.extend(held); all_oracles.extend(oracle); canonical_rows.extend(canonical)
    all_same, all_held, all_oracles = np.asarray(all_same), np.asarray(all_held), np.asarray(all_oracles)
    averaged = np.column_stack([all_same.mean(axis=1), all_held.mean(axis=1), all_oracles])
    out['macro_summaries'] = {}; out['canonical_single_split'] = dict(seed=0, rng_scope=out['meta']['canonical_rng_scope'])
    for label, estimand, s, h in [('task_equal', 'task_equal_passage', np.mean(task_same, axis=0), np.mean(task_held, axis=0)),
                                 ('context_pooled', 'passage_equal', all_same.mean(axis=0), all_held.mean(axis=0))]:
        macro = split_summary(s, h)
        sampled = clustered_summary(averaged, all_pids, all_tasks, estimand=estimand, task_order=TASKS)
        macro.update(same_passage_ci=sampled['ci'][0], held_passage_ci=sampled['ci'][1],
                     ctx_oracle=sampled['point'][2], ctx_oracle_ci=sampled['ci'][2], estimand=estimand, n_ctx=len(all_pids))
        out['macro_summaries'][label] = macro
        single = clustered_summary(canonical_rows, all_pids, all_tasks, estimand=estimand, task_order=TASKS)
        out['canonical_single_split'][label] = dict(same=single['point'][0], held=single['point'][1],
                                                   same_ci=single['ci'][0], held_ci=single['ci'][1], estimand=estimand, n_ctx=len(all_pids))
    equal = out['macro_summaries']['task_equal']
    out['macro'] = {k: equal[k] for k in ['same', 'held', 'ctx_oracle']}
    for field in ['C', 'top1', 'signconf', 'retention']:
        out['macro'][field] = float(np.mean([out['tasks'][t][field] for t in TASKS]))
    out['task_level'] = dict(rho_C_retention=safe_correlation([out['tasks'][t]['C'] for t in TASKS], [out['tasks'][t]['retention'] for t in TASKS]),
                             rho_C_ctxoracle=safe_correlation([out['tasks'][t]['C'] for t in TASKS], [out['tasks'][t]['ctx_oracle'] for t in TASKS]))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('writer', nargs='?', choices=['qwen', 'gemma'], default='qwen'); p.add_argument('--output-dir')
    args = p.parse_args(); results = os.environ.get('RESULTS_DIR', 'results')
    out = analyse_writer(args.writer, results)
    directory = Path(args.output_dir or str(Path(results) / 'q1_query_consistency')); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{args.writer}.json'; path.write_text(json.dumps(out, indent=2, allow_nan=False))
    for label, values in out['macro_summaries'].items():
        print(args.writer, label, 'held=%+.6f SD=%.6f positive=%.2f' % (values['held'], values['held_sd'], values['positive_fraction']))
    print(args.writer, 'canonical', json.dumps(out['canonical_single_split'])); print('WROTE', path)

if __name__ == '__main__': main()
