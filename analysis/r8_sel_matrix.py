#!/usr/bin/env python3
"""R8 Q3: selector capacity control under the EXACT G4 P3 protocol (Qwen).

Arms read the SAME pools (m0a_qwen rank8 seeds 42..49), SAME rng=0 context thirds,
SAME train/dev/test action seeds (42-44 / 45-46 / 47-49), SAME dev-frozen coverage-0.5 gate,
SAME selected-minus-baseline metric, 3 critic init seeds. Only the selector capacity varies.

Arms:
  current    : 2-layer d256 Transformer (reference; should reproduce g4_deployreal/qwen_ctx)
  stronger   : 4-layer d512 Transformer, same mu+action input

Query-aware diagnostic is NOT run: the fixed pools log no per-query text/embedding, so a real
query channel cannot be constructed without a new search. We state that honestly in the paper.
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MU_DIR = os.path.join(os.environ.get("SCRIPTS_DIR", "scripts"), "qwen_mu_tokens")
L_LAYERS, R_RANK, D_MODEL = 36, 8, 512
N_TOKENS = L_LAYERS * R_RANK
POOL = os.path.join(os.environ.get("RESULTS_DIR", "results"), "m0a_qwen", "rank8_seedSEED_TASK.json")
TASKS = ["squad", "drop", "ropes", "pwc"]


def gate_dev(act):
    a = torch.as_tensor(np.asarray(act), dtype=torch.float32); K = a.shape[0]
    return a[:, None, :].expand(K, L_LAYERS, R_RANK).reshape(K, N_TOKENS, 1)


class DS(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.entries = entries; self._mu = {}
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
        return {"mu": self._mu_tok(e), "u": gate_dev(e["act"]),
                "dr": torch.tensor(e["delta_r"], dtype=torch.float32),
                "task": e["task"], "row": e["row"]}


def collate(b):
    return {"mu": torch.stack([x["mu"] for x in b]), "u": torch.stack([x["u"] for x in b]),
            "dr": torch.stack([x["dr"] for x in b]), "task": [x["task"] for x in b],
            "row": [x["row"] for x in b]}


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
    out = []
    for task in TASKS:
        f = POOL.replace("SEED", str(seed)).replace("TASK", task)
        if not os.path.isfile(f): continue
        d = json.load(open(f)); act = np.load(f.replace(".json", ".actions.npz"))["actions"]
        for i, r in enumerate(d["rows"]):
            out.append({"task": task, "row": int(r["row"]), "seed": seed,
                        "baseline_reward": float(r["baseline_reward"]),
                        "delta_r": (np.asarray(r["candidate_rewards"], dtype=np.float32) - float(r["baseline_reward"])).tolist(),
                        "act": act[i]})
    return out


@torch.no_grad()
def predict_scores(model, entries, device):
    ld = torch.utils.data.DataLoader(DS(entries), batch_size=8, shuffle=False, collate_fn=collate)
    sel_delta, max_score, all_dr = [], [], []
    for b in ld:
        A = model(b["mu"].to(device), b["u"].to(device)).cpu()
        dr = b["dr"]; sel = A.argmax(1); rows = range(dr.shape[0])
        sel_delta += dr[rows, sel].numpy().tolist()
        max_score += A.max(1).values.numpy().tolist()
        all_dr += dr.numpy().tolist()
    return np.array(sel_delta), np.array(max_score), np.array(all_dr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["current", "stronger"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = "cuda"
    tr_seeds, dv_seeds, te_seeds = [42, 43, 44], [45, 46], [47, 48, 49]
    train_pool = sum([load_seed(s) for s in tr_seeds], [])
    dev_pool = sum([load_seed(s) for s in dv_seeds], [])
    test_pool = sum([load_seed(s) for s in te_seeds], [])
    all_rows = sorted({e["row"] for e in train_pool} | {e["row"] for e in dev_pool} | {e["row"] for e in test_pool})
    rng = np.random.default_rng(0); perm = rng.permutation(len(all_rows))
    n = len(all_rows); a1, a2 = n // 3, 2 * n // 3
    tr_rows = set(all_rows[i] for i in perm[:a1]); dv_rows = set(all_rows[i] for i in perm[a1:a2]); te_rows = set(all_rows[i] for i in perm[a2:])
    train = [e for e in train_pool if e["row"] in tr_rows]
    dev = [e for e in dev_pool if e["row"] in dv_rows]
    test = [e for e in test_pool if e["row"] in te_rows]
    cfg = dict(current=dict(d_hidden=256, n_layers=2),
               stronger=dict(d_hidden=512, n_layers=4))[a.arm]
    cells = []
    for init in [0, 1, 2]:
        torch.manual_seed(init)
        model = Critic(cfg["d_hidden"], cfg["n_layers"], 4).to(device)
        ds_tr = DS(train); acc, nseen = torch.zeros(N_TOKENS, 1, device=device), 0
        for i in range(0, min(len(ds_tr), 256)):
            acc += ds_tr[i]["mu"].to(device).pow(2).mean(-1, keepdim=True); nseen += 1
        model.set_mu_norm((acc / max(nseen, 1)).sqrt())
        ld_tr = torch.utils.data.DataLoader(DS(train), batch_size=8, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        for ep in range(20):
            model.train()
            for b in ld_tr:
                A = model(b["mu"].to(device), b["u"].to(device)); dr = b["dr"].to(device)
                L = loss_reg(A, dr) + loss_rank(A, dr, 1e-9)[0]
                opt.zero_grad(set_to_none=True); L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        dv_sel, dv_gate, _ = predict_scores(model, dev, device)
        best_thr, best_net = None, -1e9
        for cov in [0.3, 0.4, 0.5, 0.6, 0.7]:
            thr = np.quantile(dv_gate, 1 - cov); mask = dv_gate > thr
            net = np.where(mask, dv_sel, 0.0).mean()
            if net > best_net: best_net, best_thr = net, thr
        te_sel, te_gate, te_all = predict_scores(model, test, device)
        mask = te_gate > best_thr
        te_gated = np.where(mask, te_sel, 0.0)
        def ci(x):
            n = len(x); b = x[rng.integers(0, n, (10000, n))].mean(1)
            return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        per_task = {}
        for t in TASKS:
            idx = np.array([i for i, e in enumerate(test) if e["task"] == t])
            if len(idx): per_task[t] = float(te_sel[idx].mean())
        cells.append({"init": init, "sel": float(te_sel.mean()), "ci_sel": ci(te_sel),
                      "gated": float(te_gated.mean()), "ci_gated": ci(te_gated),
                      "random": float(te_all.mean(1).mean()), "oracle": float(te_all.max(1).mean()),
                      "coverage": float(mask.mean()), "per_task": per_task})
        print(json.dumps(cells[-1]), flush=True)
    out = {"arm": a.arm, "cfg": cfg,
           "meta": {"train_seeds": tr_seeds, "dev_seeds": dv_seeds, "test_seeds": te_seeds,
                    "n_train": len(train), "n_dev": len(dev), "n_test": len(test)},
           "init_cells": cells}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
