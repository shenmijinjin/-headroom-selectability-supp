"""Targeted statistical regressions; run: python3 -m unittest discover -s analysis -p 'test_*.py'."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from bootstrap_utils import averaged_prediction_rows, clustered_summary
from p3_statistics import summarize_inits, validated_pool
from r41_factorial_ci import load_raw, summarize_factorial
from r42_querycount import passage_comparison
from r8_q1_consistency import split_summary, fixed_passage_stats
from build_paper_summary import legacy_p3


class BootstrapTests(unittest.TestCase):
    def test_repeated_mc_rows_are_not_new_passages(self):
        # This fails under the former flatten(n_passages*20) bootstrap.
        values = np.array([-2., -.5, .1, .6, 1.4, 3.])
        ids = ["p" + str(i) for i in range(len(values))]
        for estimand in ["passage_equal", "row_pooled"]:
            original = clustered_summary(values, ids, estimand=estimand, n_boot=2000, seed=7)
            repeated = clustered_summary(np.repeat(values,20), np.repeat(ids,20), estimand=estimand, n_boot=2000,seed=7)
            self.assertEqual(original["n_passages"], repeated["n_passages"])
            np.testing.assert_allclose(original["ci"], repeated["ci"], atol=1e-14,rtol=0)
            self.assertAlmostEqual(original["point"], repeated["point"])

    def test_estimands_and_bootstrap_match_manual_row_weighting(self):
        values = np.array([0.,0.,9.,100.])
        ids = ["a1","a1","a2","b1"]
        tasks = ["a","a","a","b"]
        result = clustered_summary(values,ids,tasks,estimand="task_equal_row",n_boot=100,seed=11,return_samples=True)
        self.assertEqual(result["point"],51.5)
        # Bootstrap a's two passages together with all their rows, preserving
        # passage row counts; b has one passage. Exact same seeded draw schedule.
        draws = np.random.default_rng(11).integers(0,2,(100,2))
        a1 = (draws == 0).sum(1); a2 = (draws == 1).sum(1)
        expected = ((9*a2)/(2*a1+a2)+100)/2
        np.testing.assert_allclose(result["bootstrap_samples"],expected)
        np.testing.assert_allclose(result["ci"],np.quantile(expected,[.025,.975]))
        self.assertEqual(clustered_summary(values,ids,tasks,estimand="task_equal_passage",n_boot=10)["point"],52.25)
        self.assertEqual(clustered_summary(values,ids,tasks,estimand="row_pooled",n_boot=10)["point"],27.25)

    def test_shared_columns_keep_paired_constant_contrast(self):
        x = np.arange(10,dtype=float)
        result = clustered_summary(np.column_stack([x,x+.25]),list(map(str,range(10))),n_boot=1000,return_samples=True)
        np.testing.assert_allclose(np.diff(result["bootstrap_samples"],axis=1),.25,atol=1e-14)

    def test_cross_task_and_split_passage_rejected(self):
        with self.assertRaises(ValueError):
            clustered_summary([1.,2.],["p","p"],["a","b"],n_boot=10)
        with self.assertRaises(ValueError):
            clustered_summary([1.,2.],["p","p"],strata=["seen","unseen"],n_boot=10)
        with self.assertRaises(ValueError):
            clustered_summary([1.,2.],["p","q"],["a","a"],task_order=["a","b"],n_boot=10)

    def test_init_alignment_rejects_missing_duplicate_and_nonfinite(self):
        row = dict(task="a",passage_id="p",row=0,action_seed=42,selected=1.)
        with self.assertRaises(ValueError):
            averaged_prediction_rows([[row,row]],["selected"])
        with self.assertRaises(ValueError):
            averaged_prediction_rows([[row],[]],["selected"])
        with self.assertRaises(ValueError):
            averaged_prediction_rows([[dict(row,selected=float("nan"))]],["selected"])


class FixedFitTests(unittest.TestCase):
    def test_required_pool_files_and_candidate_counts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            with self.assertRaises(FileNotFoundError):
                validated_pool(path)
            path.write_text(json.dumps({"rows": [dict(row=0, context="passage", baseline_reward=0., candidate_rewards=[0., 1.])]}))
            with self.assertRaises(FileNotFoundError):
                validated_pool(path)
            np.savez(path.with_suffix(".actions.npz"), actions=np.zeros((1,3,8)))
            with self.assertRaisesRegex(ValueError, "Candidate"):
                validated_pool(path)
            np.savez(path.with_suffix(".actions.npz"), actions=np.zeros((2,2,8)))
            with self.assertRaisesRegex(ValueError, "row counts"):
                validated_pool(path)
            np.savez(path.with_suffix(".actions.npz"), actions=np.zeros((1,2,8)))
            data, action = validated_pool(path)
            self.assertEqual(len(data["rows"]), 1)
            self.assertEqual(action.shape, (1,2,8))

    def p3_rows(self):
        rows=[]
        for ti,task in enumerate(["squad","drop","ropes","pwc"]):
            for pi in range(4):
                for row in range(pi+1):
                    value=float(ti+pi+row)/10
                    rows.append(dict(task=task,passage_id=f"{task}/{pi}",row=row,action_seed=47,
                                     selected=value,gated=value/2,random=-.1,oracle=value+.2,coverage=.5))
        return rows

    def test_duplicate_critic_init_does_not_shrink_p3_ci(self):
        rows=self.p3_rows()
        single=summarize_inits([dict(init=0,rows=rows)],n_boot=500)
        triple=summarize_inits([dict(init=i,rows=list(reversed(rows))) for i in range(3)],n_boot=500)
        for estimand in ["task_equal_row","row_pooled"]:
            for metric in single[estimand]["metrics"]:
                for field in ["point", "ci"]:
                    np.testing.assert_allclose(single[estimand]["metrics"][metric][field],
                                               triple[estimand]["metrics"][metric][field],atol=1e-14,rtol=0)

    def factorial_raw(self):
        rows=[]
        for ti,task in enumerate(["squad","drop","ropes","pwc"]):
            for cohort,prefix in [("seen","s"),("unseen","u")]:
                for pi in range(4):
                    for ri in range(pi+1):
                        base=(ti+pi+ri)/10
                        for cell,delta,action_seed in ([("A",0.,42),("B",.5,47)] if cohort=="seen" else [("C",0.,42),("D",.7,47)]):
                            rows.append(dict(task=task,passage_id=f"{task}/{prefix}{pi}",row=ri,
                                             action_seed=action_seed,cell=cell,selected=base+delta,random=-.2))
        return {"init_predictions":[{"init":0,"rows":rows}]}

    def test_factorial_joint_fit_and_cell_resampling(self):
        data=self.factorial_raw()
        single=summarize_factorial(data,n_boot=500)
        triple=summarize_factorial({"init_predictions":[dict(init=i,rows=list(reversed(data["init_predictions"][0]["rows"]))) for i in range(3)]},n_boot=500)
        for estimand in ["task_equal_row","row_pooled"]:
            for cell in single[estimand]["cells"]:
                for metric in ["selected_minus_base", "selected_minus_random"]:
                    for field in ["point", "ci"]:
                        np.testing.assert_allclose(single[estimand]["cells"][cell][metric][field],
                                                   triple[estimand]["cells"][cell][metric][field],atol=1e-14,rtol=0)
            np.testing.assert_allclose(single[estimand]["contrasts"]["A-B"]["ci"],[-.5,-.5],atol=1e-14)
            np.testing.assert_allclose(single[estimand]["contrasts"]["C-D"]["ci"],[-.7,-.7],atol=1e-14)

    def test_factorial_summary_cannot_manufacture_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json"
            path.write_text(json.dumps({"init_cells":[{"A":{"sel_minus_base":.03}}]}))
            with self.assertRaisesRegex(ValueError,"scalar summaries"):
                load_raw(path)

    def test_legacy_p3_macro_is_exact_but_gated_macro_ci_unavailable(self):
        cells=[dict(init=i,per_task=dict(squad=.1,drop=.2,ropes=.3,pwc=.4),
                    sel=.99,gated=.01,coverage=.5,random=0.,oracle=1.) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.json";path.write_text(json.dumps(dict(meta={},init_cells=cells)))
            result=legacy_p3(path)
        self.assertAlmostEqual(result["no_gate_task_equal"]["point"],.25)
        self.assertIsNone(result["no_gate_task_equal"]["ci"])
        self.assertEqual(result["gate_row_pooled"]["point"],.01)
        self.assertIsNone(result["gate_row_pooled"]["task_equal_point"])
        self.assertIsNone(result["gate_row_pooled"]["ci"])


class QueryAccessTests(unittest.TestCase):
    def test_common_query_eval_and_mc_aggregation(self):
        cr=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
        br=np.zeros(4)
        value=passage_comparison(cr,br,20)
        self.assertAlmostEqual(value[2],value[1]-value[0])
        manual=[]
        for seed in range(1000,1020):
            perm=np.random.default_rng(seed).permutation(4)
            ev=perm[2:]
            g=[]
            for k in [1,2]:
                action=cr[:,perm[:k]].mean(1).argmax()
                g.append(cr[action,ev].mean())
            manual.append([g[0],g[1],g[1]-g[0]])
        np.testing.assert_allclose(value,np.mean(manual,axis=0))

    def test_partition_sd_positivity_and_ratio_use_actual_macro_values(self):
        result=split_summary([1.,2.,3.,4.],[-1.,0.,1.,2.])
        self.assertEqual(result["positive_fraction"],.5)
        self.assertEqual(result["held"],.5)
        self.assertAlmostEqual(result["held_sd"],np.std([-1.,0.,1.,2.],ddof=0))
        self.assertEqual(result["retention_ratio_of_means"],.2)

    def test_sign_conflict_is_symmetric_and_zeros_are_not_opposite(self):
        response=np.array([[1.,-1.],[-1.,1.],[0.,1.],[1.,0.]])
        self.assertEqual(fixed_passage_stats(response)["signconf"],.5)
        self.assertEqual(fixed_passage_stats(response[:,::-1])["signconf"],.5)


if __name__ == "__main__":
    unittest.main()
