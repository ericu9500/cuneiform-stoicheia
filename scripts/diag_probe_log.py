"""Probe 1 (log part): train-loss trend and dev-curve noise from the base-long training log / history.
Torch-free.  python scripts/diag_probe_log.py"""
import json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "diag", "probes")
rx = re.compile(r"step (\d+) lr (\S+) loss (\S+) sign (\S+) read (\S+) join (\S+) det (\S+) (\d+) steps/100s")
rows = [tuple(float(x) for x in m.groups()) for m in map(rx.search, open(os.path.join(ROOT, "logs", "cun-base-long-5492605.log"))) if m]
tr = np.array(rows)
done = json.load(open(os.path.join(ROOT, "ckpt", "base-long", "done.json")))
h = done["history"]
dev = np.array([[x["step"], x["dev_loss"], x["dev_acc_sign"], x["dev_acc_read"], x["lr"]] for x in h])
res = {"anneal_start": done["anneal_start"], "steps": done["steps"], "n_train_points": len(tr)}
print("train points", len(tr), "dev evals", len(dev), "anneal", done["anneal_start"])

# train loss (single-batch values, logged every 100 steps) in windows
print("\nwindow        n   total  (se)    sign   read   join   det   | dev loss (mean of evals in window)")
win = []
edges = list(range(0, 76001, 4000))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (tr[:, 0] > lo) & (tr[:, 0] <= hi)
    d = dev[(dev[:, 0] > lo) & (dev[:, 0] <= hi)]
    x = tr[m]
    w = {"lo": lo, "hi": hi, "n": int(m.sum()), "loss": x[:, 2].mean(), "se": x[:, 2].std(ddof=1) / np.sqrt(m.sum()),
         "sign": x[:, 3].mean(), "read": x[:, 4].mean(), "join": x[:, 5].mean(), "det": x[:, 6].mean(),
         "dev_loss": d[:, 1].mean(), "dev_sign_acc": d[:, 2].mean()}
    win.append(w)
    print(f"{lo:6d}-{hi:6d} {w['n']:3d}  {w['loss']:.3f} ({w['se']:.3f})  {w['sign']:.3f}  {w['read']:.3f}  {w['join']:.3f}  {w['det']:.3f} | {w['dev_loss']:.4f}  acc {w['dev_sign_acc']:.4f}")
res["windows"] = win
res["single_batch_sd"] = float(tr[tr[:, 0] > 60000, 2].std(ddof=1))

# linear trend of the train loss per 10k steps in phases
def slope(lo, hi, col=2):
    m = (tr[:, 0] > lo) & (tr[:, 0] <= hi)
    x, y = tr[m, 0] / 1e4, tr[m, col]
    A = np.c_[x, np.ones_like(x)]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    se = np.sqrt(r.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return float(beta[0]), float(se)

def dslope(lo, hi):
    m = (dev[:, 0] > lo) & (dev[:, 0] <= hi)
    x, y = dev[m, 0] / 1e4, dev[m, 1]
    A = np.c_[x, np.ones_like(x)]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    se = np.sqrt(r.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return float(beta[0]), float(se), float(r.std(ddof=2))

print("\nslopes per 10k steps (train total loss; train sign loss; dev loss; SD of dev residuals about the line)")
res["slopes"] = {}
for lo, hi in ((20000, 30000), (30000, 46000), (36000, 46000), (46000, 56000), (56000, 66000), (66000, 76000)):
    a, b, c = slope(lo, hi), slope(lo, hi, 3), dslope(lo, hi)
    res["slopes"][f"{lo}-{hi}"] = {"train": a, "train_sign": b, "dev": c[:2], "dev_resid_sd": c[2]}
    print(f"{lo:6d}-{hi:6d}  train {a[0]:+.4f} ({a[1]:.4f})  sign {b[0]:+.4f} ({b[1]:.4f})  dev {c[0]:+.4f} ({c[1]:.4f})  resid SD {c[2]:.4f}")

# eval-to-eval noise of the dev curve at peak LR: successive differences, detrended
for lo, hi in ((20000, 46000), (30000, 46000), (60000, 76000)):
    m = (dev[:, 0] > lo) & (dev[:, 0] <= hi)
    y = dev[m, 1]; a = dev[m, 2]
    d = np.diff(y)
    sd = float(np.std(d - d.mean(), ddof=1) / np.sqrt(2))
    sda = float(np.std(np.diff(a) - np.diff(a).mean(), ddof=1) / np.sqrt(2))
    res[f"dev_noise_{lo}-{hi}"] = {"n_evals": int(m.sum()), "loss_sd": sd, "sign_acc_sd": sda,
                                   "n_increases": int((d > 0).sum())}
    print(f"dev eval-to-eval noise {lo}-{hi}: n {int(m.sum())}  loss SD {sd:.4f}  sign-acc SD {sda:.4f}  increases {int((d > 0).sum())}/{len(d)}")
print("\ndev history around the anneal:")
for x in h:
    if 38000 <= x["step"] <= 50000 or x["step"] >= 70000:
        print(x["step"], f"{x['lr']:.2e}", round(x["dev_loss"], 4), round(x["dev_acc_sign"], 4), round(x["dev_acc_read"], 4))
res["dev_at"] = {str(int(s)): float(v) for s, v in dev[:, :2] if s in (10000, 20000, 30000, 40000, 46000, 50000, 60000, 70000, 76000)}
res["positions_seen"] = 76000 * 64 * 512 * 0.97
json.dump(res, open(os.path.join(OUT, "log_trend.json"), "w"), indent=1)
