#!/usr/bin/env python3
"""R41 factorial: passage-level 2x2 (context x action) decomposition, Qwen + Gemma.
Same critic/protocol as r41_g4_passage but splits passages into train/test (dev merged for the 2x2,
no gate) and evaluates the four cells. No gate calibration."""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
TASKS = ["squad", "drop", "ropes", "pwc"]

def make(L_LAYERS, D_MODEL, MU_DIR, POOL):
    R_RANK = 8; N_TOKENS = L_LAYERS * R_RANK
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
                    "dr": torch.tensor(e["delta_r"], dtype=torch.float32), "task": e["task"]}
    def collate(b):
        return {"mu": torch.stack([x["mu"] for x in b]), "u": torch.stack([x["u"] for x in b]),
                "dr": torch.stack([x["dr"] for x in b]), "task": [x["task"] for x in b]}
    class Critic(nn.Module):
        def __init__(self, d_hidden=256, n_layers=2, n_head=4):
            super().__init__()
            self.inp = nn.Linear(D_MODEL + 1, d_hidden); self.pos = nn.Parameter(torch.zeros(N_TOKENS, d_hidden))
            l = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=n_head, dim_feedforward=4*d_hidden, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(l, num_layers=n_layers); self.norm = nn.LayerNorm(d_hidden)
            self.head = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
            self.register_buffer("mu_rms", torch.ones(N_TOKENS, 1))
        def set_mu_norm(self, m): self.mu_rms.copy_(m.clamp_min(1e-6))
        def forward(self, mu, u):
            B, K = u.shape[0], u.shape[1]
            mu_n = (mu / self.mu_rms).unsqueeze(1).expand(B, K, N_TOKENS, D_MODEL)
            x = torch.cat([mu_n, u], -1).reshape(B*K, N_TOKENS, -1)
            h = self.inp(x) + self.pos; h = self.norm(self.enc(h)).mean(1); return self.head(h).reshape(B, K)
    def load_seed(seed):
        out = []
        for task in TASKS:
            f = POOL.replace("SEED", str(seed)).replace("TASK", task)
            if not os.path.isfile(f): continue
            d = json.load(open(f)); act = np.load(f.replace(".json", ".actions.npz"))["actions"]
            for i, r in enumerate(d["rows"]):
                out.append({"task": task, "row": int(r["row"]), "pid": "%s::%s" % (task, r["context"]),
                            "delta_r": (np.asarray(r["candidate_rewards"], dtype=np.float32) - float(r["baseline_reward"])).tolist(),
                            "act": act[i]})
        return out
    def loss_fn(A, dr):
        L = F.mse_loss(A, dr)
        ai, aj = A.unsqueeze(2), A.unsqueeze(1); ri, rj = dr.unsqueeze(2), dr.unsqueeze(1)
        keep = (ri - rj) > 1e-9
        if keep.any(): L = L - F.logsigmoid((ai - aj)[keep]).mean()
        return L
    @torch.no_grad()
    def predict(model, entries, device):
        ld = torch.utils.data.DataLoader(DS(entries), batch_size=8, shuffle=False, collate_fn=collate)
        sd, allr = [], []
        for b in ld:
            A = model(b["mu"].to(device), b["u"].to(device)).cpu(); dr = b["dr"]
            sel = A.argmax(1); rows = range(dr.shape[0])
            sd += dr[rows, sel].numpy().tolist(); allr += dr.numpy().tolist()
        return np.array(sd), np.array(allr)
    return DS, Critic, load_seed, loss_fn, predict, collate

def run(name, L_LAYERS, MU_DIR, POOL, tr_seeds, te_seeds, out_dir):
    DS, Critic, load_seed, loss_fn, predict, collate = make(L_LAYERS, 512, MU_DIR, POOL)
    device = "cuda"; os.makedirs(out_dir, exist_ok=True)
    train_pool = sum([load_seed(s) for s in tr_seeds], [])
    test_pool = sum([load_seed(s) for s in te_seeds], [])
    all_pids = sorted({e["pid"] for e in train_pool} | {e["pid"] for e in test_pool})
    rng0 = np.random.default_rng(0); perm = rng0.permutation(len(all_pids)); n = len(all_pids)
    a1 = n // 2
    tr_p = set(all_pids[i] for i in perm[:a1]); te_p = set(all_pids[i] for i in perm[a1:])
    cell_A = [e for e in train_pool if e["pid"] in tr_p]
    cell_B = [e for e in test_pool if e["pid"] in tr_p]
    cell_C = [e for e in train_pool if e["pid"] in te_p]
    cell_D = [e for e in test_pool if e["pid"] in te_p]
    print(name, "cell sizes A/B/C/D:", len(cell_A), len(cell_B), len(cell_C), len(cell_D), flush=True)
    results = {"meta": {"name": name, "split_unit": "passage", "train_seeds": tr_seeds, "test_seeds": te_seeds}, "init_cells": []}
    rng = np.random.default_rng(7)
    for init in [0, 1, 2]:
        torch.manual_seed(init); model = Critic().to(device)
        ds_tr = DS(cell_A); acc, nseen = torch.zeros(DS(cell_A)[0]["mu"].shape[0], 1, device=device), 0
        for i in range(0, min(len(ds_tr), 256)):
            acc += ds_tr[i]["mu"].to(device).pow(2).mean(-1, keepdim=True); nseen += 1
        model.set_mu_norm((acc / max(nseen, 1)).sqrt())
        ld_tr = torch.utils.data.DataLoader(DS(cell_A), batch_size=8, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        for ep in range(20):
            model.train()
            for b in ld_tr:
                A = model(b["mu"].to(device), b["u"].to(device)); dr = b["dr"].to(device)
                L = loss_fn(A, dr); opt.zero_grad(set_to_none=True); L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); cell = {"init": init}
        for nm, entries in [("A", cell_A), ("B", cell_B), ("C", cell_C), ("D", cell_D)]:
            sd, allr = predict(model, entries, device)
            cell[nm] = {"sel_minus_random": float((sd - allr.mean(1)).mean()), "sel_minus_base": float(sd.mean())}
            print(json.dumps({"init": init, "cell": nm, "sel_base": cell[nm]["sel_minus_base"]}), flush=True)
        results["init_cells"].append(cell)
    json.dump(results, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
    print("WROTE", name, flush=True)

if __name__ == "__main__":
    R = os.environ.get("RESULTS_DIR", "results")
    S = os.environ.get("SCRIPTS_DIR", "scripts")
    SFA = os.environ.get("SFA_DIR", "sfa")
    run("qwen", 36, S + "/qwen_mu_tokens", R + "/m0a_qwen/rank8_seedSEED_TASK.json",
        [42,43,44], [47,48,49], R + "/e1_factorial_passage_qwen")
    run("gemma", 26, SFA + "/mu_tokens",
        R + "/m0a/rank8_seedSEED_TASK.json", [42,43], [45], R + "/e1_factorial_passage_gemma")
