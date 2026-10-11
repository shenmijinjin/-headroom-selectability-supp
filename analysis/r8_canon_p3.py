#!/usr/bin/env python3
"""R8 Q2 (fixed): canonicalized-action P3 vs native-gate P3 on the E4 shared-action subset.

delta_r in both files is ALREADY relative to baseline (verified: mean native +0.008, canon -0.024).
The SAME (task, row) pair must stay in ONE split across all four tasks; we split on the 24 shared
row ids (rng=0 thirds), then take every task's context at that row into the same partition.
Critic input = mu tokens + gate-deviation of the 8-d action. Metric = selected - baseline.
Native vs canonical differ ONLY in which reward matrix the critic trains/evaluates on.
"""
import json, os, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

MU_DIR = os.path.join(os.environ.get("SCRIPTS_DIR", "scripts"), "qwen_mu_tokens")
L_LAYERS, R_RANK, D_MODEL = 36, 8, 512
N_TOKENS = L_LAYERS * R_RANK
BASE = os.environ.get("RESULTS_DIR", "results")
TASKS = ["squad", "drop", "ropes", "pwc"]


def gate_dev(act8):
    a = torch.as_tensor(np.asarray(act8, dtype=np.float32))
    return a[None, :].expand(L_LAYERS, R_RANK).reshape(N_TOKENS, 1)


def load_cells(kind):
    out = []
    for task in TASKS:
        f = ("%s/e4_landscape/qwen_%s.json" % (BASE, task)) if kind == "native" else \
            ("%s/e4_canonical/rank8_canon_%s.json" % (BASE, task))
        d = json.load(open(f))
        for c in d["contexts"]:
            out.append({"task": task, "row": int(c["row"]),
                        "dr": np.asarray(c["delta_r"], dtype=np.float32).tolist()})  # already delta
    return out


_mu_cache = {}
def mu_tok(task, row):
    t = _mu_cache.get(task)
    if t is None:
        z = np.load(os.path.join(MU_DIR, "%s.pooled.npz" % task))
        t = {int(k.split("_")[1]): torch.from_numpy(np.asarray(z[k], dtype=np.float32)) for k in z.files}
        _mu_cache[task] = t
    return t[row]


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


def loss_fn(A, dr):
    L = F.mse_loss(A, dr)
    ai, aj = A.unsqueeze(2), A.unsqueeze(1)
    ri, rj = dr.unsqueeze(2), dr.unsqueeze(1)
    keep = (ri - rj) > 1e-9
    if keep.any():
        L = L - F.logsigmoid((ai - aj)[keep]).mean()
    return L


def run_arm_split(cells, tr_rows, tr_act, dv_rows, dv_act, te_rows, te_act, device, init):
    torch.manual_seed(init)
    model = Critic().to(device)
    tr = [e for e in cells if e["row"] in tr_rows]
    dv = [e for e in cells if e["row"] in dv_rows]
    te = [e for e in cells if e["row"] in te_rows]
    acc, nseen = torch.zeros(N_TOKENS, 1, device=device), 0
    for e in tr[:256]:
        acc += mu_tok(e["task"], e["row"]).to(device).pow(2).mean(-1, keepdim=True); nseen += 1
    model.set_mu_norm((acc / max(nseen, 1)).sqrt())

    def make(entries, act_idx, shuffle):
        idx = np.random.permutation(len(entries)) if shuffle else np.arange(len(entries))
        for i in range(0, len(idx), 8):
            sel = [entries[j] for j in idx[i:i+8]]
            mu = torch.stack([mu_tok(e["task"], e["row"]) for e in sel]).to(device)
            u = torch.stack([torch.stack([gate_dev(ACTIONS[k]) for k in act_idx]) for _ in sel]).to(device)
            dr = torch.tensor([[e["dr"][k] for k in act_idx] for e in sel], dtype=torch.float32).to(device)
            yield mu, u, dr
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for ep in range(20):
        model.train()
        for mu, u, dr in make(tr, tr_act, True):
            L = loss_fn(model(mu, u), dr)
            opt.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    @torch.no_grad()
    def eval_sel(entries, act_idx):
        sd, mx, allr = [], [], []
        for mu, u, dr in make(entries, act_idx, False):
            A = model(mu, u)
            sd += dr.gather(1, A.argmax(1, keepdim=True)).squeeze(1).cpu().tolist()
            mx += A.max(1).values.cpu().tolist()
            allr += dr.cpu().tolist()
        return np.array(sd), np.array(mx), np.array(allr)
    dv_sel, dv_gate, _ = eval_sel(dv, dv_act)
    best_thr, best_net = None, -1e9
    for cov in [0.3, 0.4, 0.5, 0.6, 0.7]:
        thr = np.quantile(dv_gate, 1 - cov); mask = dv_gate > thr
        net = np.where(mask, dv_sel, 0.0).mean()
        if net > best_net: best_net, best_thr = net, thr
    te_sel, te_gate, te_all = eval_sel(te, te_act)
    mask = te_gate > best_thr
    te_gated = np.where(mask, te_sel, 0.0)
    rng = np.random.default_rng(0)
    def ci(x):
        n = len(x)
        if n < 3: return [float("nan"), float("nan")]
        b = x[rng.integers(0, n, (10000, n))].mean(1)
        return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    return dict(sel=float(te_sel.mean()), ci_sel=ci(te_sel), gated=float(te_gated.mean()),
                ci_gated=ci(te_gated), oracle=float(te_all.max(1).mean()),
                random=float(te_all.mean()), coverage=float(mask.mean()), n_test=len(te),
                per_task={t: float(np.mean([te_sel[i] for i, e in enumerate(te) if e["task"] == t]))
                          for t in TASKS})


_d = json.load(open("%s/e4_landscape/qwen_squad.json" % BASE))
ACTIONS = [np.asarray(a, dtype=np.float32) for a in _d["shared_actions"]]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native = load_cells("native"); canon = load_cells("canon")
    # split on the 24 shared row ids; same row -> same split across all tasks
    rows = sorted({e["row"] for e in native})
    rng = np.random.default_rng(0); perm = rng.permutation(len(rows))
    n = len(rows); a1, a2 = n // 3, 2 * n // 3
    tr_rows = set(rows[i] for i in perm[:a1]); dv_rows = set(rows[i] for i in perm[a1:a2]); te_rows = set(rows[i] for i in perm[a2:])
    tr_act = list(range(0, 8)); dv_act = list(range(8, 12)); te_act = list(range(12, 16))
    print("rows tr/dv/te:", sorted(tr_rows), sorted(dv_rows), sorted(te_rows))
    print("n entries tr/dv/te:",
          sum(1 for e in native if e["row"] in tr_rows),
          sum(1 for e in native if e["row"] in dv_rows),
          sum(1 for e in native if e["row"] in te_rows))
    results = {"meta": dict(n_rows=len(rows), n_test_entries=sum(1 for e in native if e["row"] in te_rows),
                            tr_act=tr_act, dv_act=dv_act, te_act=te_act)}
    for kind, cells in [("native", native), ("canon", canon)]:
        cells_r = [run_arm_split(cells, tr_rows, tr_act, dv_rows, dv_act, te_rows, te_act, device, init)
                   for init in [0, 1, 2]]
        results[kind] = cells_r
        sels = [c["sel"] for c in cells_r]
        print(f"[{kind}] P3 sel per init = {[round(x,4) for x in sels]}  mean={np.mean(sels):+.4f}  "
              f"oracle={cells_r[0]['oracle']:+.4f} random={cells_r[0]['random']:+.4f}")
    # paired diff native vs canon per init
    diffs = [c["sel"] - n["sel"] for c, n in zip(results["canon"], results["native"])]
    print("paired canon-native sel per init:", [round(d, 4) for d in diffs], "mean=%+.4f" % np.mean(diffs))
    results["paired_canon_minus_native"] = [float(d) for d in diffs]
    out = os.path.join(os.environ.get("RESULTS_DIR", "results"), "canon_p3", "report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
