#!/usr/bin/env python3
"""R41: passage-level G4 deployment evaluation (Gemma).

Identical to r18_g4_critic.py EXCEPT the train/dev/test split unit is the PASSAGE, not the QA row.
A passage groups every QA row that shares the same tokenized context (ctx_ids). All rows of a
passage go to one split, so no test passage's representation is seen in training.

Passage identity: rows that share a passage are recovered from the e2_perquery grouping (same
tokenized ctx). Rows not in any multi-query group are their own single-row passage.

Everything else is unchanged: same pools (m0a_qwen), same train/dev/test action seeds
(42-44/45-46/47-49), same 2-layer width-256 critic, same 20-epoch budget, dev-frozen coverage-0.5
gate, selected-minus-baseline, 3 init seeds, context-clustered bootstrap CI.
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MU_DIR = os.path.join(os.environ.get("SFA_DIR", "sfa"), "mu_tokens")
L_LAYERS, R_RANK, D_MODEL = 26, 8, 512
N_TOKENS = L_LAYERS * R_RANK
TASKS = ["squad", "drop", "ropes", "pwc"]
POOL = os.path.join(os.environ.get("RESULTS_DIR", "results"), "m0a", "rank8_seedSEED_TASK.json")
E2 = os.path.join(os.environ.get("RESULTS_DIR", "results"), "e2_perquery", "qwen_TASK_s42.json")


def gate_dev(act):
    a = torch.as_tensor(np.asarray(act), dtype=torch.float32); K = a.shape[0]
    return a[:, None, :].expand(K, L_LAYERS, R_RANK).reshape(K, N_TOKENS, 1)


class DS(torch.utils.data.Dataset):
    def __init__(self, entries, cf=0):
        self.entries = entries; self.cf = cf; self._mu = {}
    def __len__(self): return len(self.entries)
    def _mu_tok(self, e):
        t = self._mu.get(e["task"])
        if t is None:
            z = np.load(os.path.join(MU_DIR, "%s.pooled.npz" % e["task"]))
            t = {int(k.split("_")[1]): torch.from_numpy(np.asarray(z[k], dtype=np.float32)) for k in z.files}
            self._mu[e["task"]] = t
        return t[e["row"]]
    def __getitem__(self, i):
        e = self.entries[i]
        mu = self._mu_tok(e)
        if self.cf: mu = torch.zeros_like(mu)
        return {"mu": mu, "u": gate_dev(e["act"]),
                "dr": torch.tensor(e["delta_r"], dtype=torch.float32), "task": e["task"],
                "pid": e["pid"]}


def collate(b):
    return {"mu": torch.stack([x["mu"] for x in b]), "u": torch.stack([x["u"] for x in b]),
            "dr": torch.stack([x["dr"] for x in b]), "task": [x["task"] for x in b],
            "pid": [x["pid"] for x in b]}


class Critic(nn.Module):
    def __init__(self, d_hidden=256, n_layers=2, n_head=4):
        super().__init__()
        self.inp = nn.Linear(D_MODEL + 1, d_hidden)
        self.pos = nn.Parameter(torch.zeros(N_TOKENS, d_hidden))
        l = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=n_head, dim_feedforward=4*d_hidden,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(l, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_hidden)
        self.head = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        self.register_buffer("mu_rms", torch.ones(N_TOKENS, 1))
    def set_mu_norm(self, m): self.mu_rms.copy_(m.clamp_min(1e-6))
    def forward(self, mu, u):
        B, K = u.shape[0], u.shape[1]
        mu_n = (mu / self.mu_rms).unsqueeze(1).expand(B, K, N_TOKENS, D_MODEL)
        x = torch.cat([mu_n, u], -1).reshape(B*K, N_TOKENS, -1)
        h = self.inp(x) + self.pos
        h = self.norm(self.enc(h)).mean(1)
        return self.head(h).reshape(B, K)


def loss_reg(a, dr): return F.mse_loss(a, dr)
def loss_rank(a, dr, eps_flat):
    ai, aj = a.unsqueeze(2), a.unsqueeze(1); ri, rj = dr.unsqueeze(2), dr.unsqueeze(1)
    keep = (ri - rj) > eps_flat
    if not keep.any(): return a.new_zeros(()), 0
    return -F.logsigmoid((ai - aj)[keep]).mean(), int(keep.sum())


def load_seed(seed):
    """Group rows by their passage (the `context` text field present in the pools)."""
    out = []
    for task in TASKS:
        f = POOL.replace("SEED", str(seed)).replace("TASK", task)
        if not os.path.isfile(f): continue
        d = json.load(open(f)); act = np.load(f.replace(".json", ".actions.npz"))["actions"]
        for i, r in enumerate(d["rows"]):
            row = int(r["row"])
            p = "%s::%s" % (task, r["context"])   # passage identity = tokenized context text
            out.append({"task": task, "row": row, "pid": p, "seed": seed,
                        "delta_r": (np.asarray(r["candidate_rewards"], dtype=np.float32) - float(r["baseline_reward"])).tolist(),
                        "act": act[i]})
    return out


@torch.no_grad()
def predict_scores(model, entries, device, cf=0):
    ld = torch.utils.data.DataLoader(DS(entries, cf), batch_size=8, shuffle=False, collate_fn=collate)
    sel_delta, max_score, all_dr, pids = [], [], [], []
    for b in ld:
        A = model(b["mu"].to(device), b["u"].to(device)).cpu()
        dr = b["dr"]; sel = A.argmax(1); rows = range(dr.shape[0])
        sel_delta += dr[rows, sel].numpy().tolist()
        max_score += A.max(1).values.numpy().tolist()
        all_dr += dr.numpy().tolist()
        pids += b["pid"]
    return np.array(sel_delta), np.array(max_score), np.array(all_dr), pids


def passage_clustered_ci(sel, pids, rng, n_boot=10000):
    """Bootstrap over passages: resample passages, take all their rows together."""
    by_p = {}
    for v, p in zip(sel, pids):
        by_p.setdefault(p, []).append(v)
    pmeans = np.array([np.mean(v) for v in by_p.values()])
    n = len(pmeans)
    b = pmeans[rng.integers(0, n, (n_boot, n))].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--init_seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--context_free", type=int, default=0)
    a = p.parse_args()
    device = "cuda"; os.makedirs(a.out_dir, exist_ok=True)
    tr_seeds, dv_seeds, te_seeds = [42, 43], [44], [45, 46, 47]
    init_seeds = [int(x) for x in a.init_seeds.split(",")]
    train_pool = sum([load_seed(s) for s in tr_seeds], [])
    dev_pool = sum([load_seed(s) for s in dv_seeds], [])
    test_pool = sum([load_seed(s) for s in te_seeds], [])
    # passage-disjoint thirds: split on unique passage ids, then take all rows of each passage
    all_pids = sorted({e["pid"] for e in train_pool} | {e["pid"] for e in dev_pool} | {e["pid"] for e in test_pool})
    rng = np.random.default_rng(0); perm = rng.permutation(len(all_pids))
    n = len(all_pids); a1, a2 = n // 3, 2 * n // 3
    tr_p = set(all_pids[i] for i in perm[:a1]); dv_p = set(all_pids[i] for i in perm[a1:a2]); te_p = set(all_pids[i] for i in perm[a2:])
    train = [e for e in train_pool if e["pid"] in tr_p]
    dev = [e for e in dev_pool if e["pid"] in dv_p]
    test = [e for e in test_pool if e["pid"] in te_p]
    assert not (tr_p & te_p) and not (dv_p & te_p) and not (tr_p & dv_p)
    n_passages = {"train": len(tr_p), "dev": len(dv_p), "test": len(te_p)}
    print("passage counts:", n_passages, "entries:", len(train), len(dev), len(test), flush=True)
    results = {"meta": {"split_unit": "passage", "context_free": bool(a.context_free), "n_passages": n_passages,
                        "train_seeds": tr_seeds, "dev_seeds": dv_seeds, "test_seeds": te_seeds,
                        "n_train": len(train), "n_dev": len(dev), "n_test": len(test)},
               "init_cells": []}
    for init in init_seeds:
        torch.manual_seed(init)
        model = Critic().to(device)
        ds_tr = DS(train, a.context_free); acc, nseen = torch.zeros(N_TOKENS, 1, device=device), 0
        for i in range(0, min(len(ds_tr), 256)):
            acc += ds_tr[i]["mu"].to(device).pow(2).mean(-1, keepdim=True); nseen += 1
        model.set_mu_norm((acc / max(nseen, 1)).sqrt())
        ld_tr = torch.utils.data.DataLoader(DS(train, a.context_free), batch_size=8, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        for ep in range(a.epochs):
            model.train()
            for b in ld_tr:
                A = model(b["mu"].to(device), b["u"].to(device)); dr = b["dr"].to(device)
                L = loss_reg(A, dr) + loss_rank(A, dr, 1e-9)[0]
                opt.zero_grad(set_to_none=True); L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        dv_sel, dv_gate, _, _ = predict_scores(model, dev, device, a.context_free)
        best_thr, best_net = None, -1e9
        for cov in [0.3, 0.4, 0.5, 0.6, 0.7]:
            thr = np.quantile(dv_gate, 1 - cov); mask = dv_gate > thr
            net = np.where(mask, dv_sel, 0.0).mean()
            if net > best_net: best_net, best_thr = net, thr
        te_sel, te_gate, te_all, te_pids = predict_scores(model, test, device, a.context_free)
        mask = te_gate > best_thr
        te_gated = np.where(mask, te_sel, 0.0)
        rngb = np.random.default_rng(0)
        per_task = {}
        for t in TASKS:
            idx = np.array([i for i, e in enumerate(test) if e["task"] == t])
            if len(idx): per_task[t] = float(te_sel[idx].mean())
        results["init_cells"].append({
            "init": init, "sel": float(te_sel.mean()),
            "ci_sel_passage": passage_clustered_ci(te_sel, te_pids, rngb),
            "gated": float(te_gated.mean()),
            "ci_gated_passage": passage_clustered_ci(te_gated, te_pids, rngb),
            "random": float(te_all.mean(1).mean()), "oracle": float(te_all.max(1).mean()),
            "coverage": float(mask.mean()), "per_task": per_task,
            "n_test_passages": len(set(te_pids))})
        print(json.dumps(results["init_cells"][-1]), flush=True)
    json.dump(results, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2)
    print("WROTE", os.path.join(a.out_dir, "report.json"))


if __name__ == "__main__":
    main()
