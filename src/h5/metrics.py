"""Task and fairness metrics.

Fairness metrics (all computed on held-out data):

* ``cf_gap``        mean |P(off|x_a) - P(off|x_b)| over counterfactual pairs.  **Primary bias score B.**
* ``cf_flip_rate``  fraction of pairs whose predicted label changes.
* ``template_fped`` / ``template_range``: on neutral templates (gold = not offensive)
  sum_g |FPR_g - FPR| (Dixon et al. 2018) and mean per-template max-min P(off).
* ``group_fped`` / ``group_fned`` on real test comments tagged with proxy groups.

Comparison quantities:

* fairness gain G(m) = B(baseline) - B(m)
* recovery = G(surgery) / G(adversarial)
* cost ratio = cost(surgery) / cost(adversarial)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

POS = 1  # index of the "offensive" class


def task_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(-1)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    if probs.shape[1] == 2:
        out["offensive_f1"] = float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0))
    return out


def pos_score(probs: np.ndarray) -> np.ndarray:
    """Probability of the 'harmful' class (offensive / negative sentiment)."""
    return probs[:, POS]


def cf_metrics(pairs: List[dict], pa: np.ndarray, pb: np.ndarray, n_boot: int = 0, seed: int = 0) -> dict:
    """pa/pb: full probability matrices for text_a / text_b of every pair."""
    if len(pairs) == 0:
        return {"n_pairs": 0}
    sa, sb = pos_score(pa), pos_score(pb)
    gap = np.abs(sa - sb)
    flip = (pa.argmax(-1) != pb.argmax(-1)).astype(float)
    out = {"n_pairs": len(pairs), "cf_gap": float(gap.mean()), "cf_flip_rate": float(flip.mean())}
    if n_boot:
        out["cf_gap_ci95"] = bootstrap_ci(gap, n_boot, seed)
        out["cf_flip_rate_ci95"] = bootstrap_ci(flip, n_boot, seed)
    by = defaultdict(list)
    for i, p in enumerate(pairs):
        by[(p["source"], p["category"])].append(i)
    out["by_category"] = {f"{s}/{c}": {"n": len(ix), "cf_gap": float(gap[ix].mean()),
                                        "cf_flip_rate": float(flip[ix].mean())}
                          for (s, c), ix in sorted(by.items())}
    # signed mean gap per (category, group) - who is pushed towards "offensive"
    sg = defaultdict(list)
    for i, p in enumerate(pairs):
        sg[(p["category"], p["group_b"])].append(sb[i] - sa[i])
        sg[(p["category"], p["group_a"])].append(sa[i] - sb[i])
    out["signed_group_shift"] = {f"{c}/{g}": float(np.mean(v)) for (c, g), v in sorted(sg.items())}
    return out


def template_metrics(fills: List[dict], probs: np.ndarray, threshold: float = 0.5) -> dict:
    if not fills:
        return {"n_fills": 0}
    df = pd.DataFrame(fills)
    df["p"] = pos_score(probs)
    df["fp"] = (df["p"] >= threshold).astype(float)
    out = {"n_fills": len(df)}
    rng = df.groupby(["template_id", "row_id"])["p"].agg(lambda s: s.max() - s.min())
    out["template_range"] = float(rng.mean())
    cats = {}
    fpeds = []
    for c, g in df.groupby("category"):
        fpr = float(g["fp"].mean())
        per = g.groupby("group")["fp"].mean()
        fped = float((per - fpr).abs().sum())
        fpeds.append(fped)
        cats[c] = {"fpr": fpr, "fped": fped, "fpr_by_group": per.round(4).to_dict(),
                   "mean_p_by_group": g.groupby("group")["p"].mean().round(4).to_dict()}
    out["template_fped"] = float(np.mean(fpeds))
    out["by_category"] = cats
    return out


def group_error_metrics(tags: List[Dict[str, str]], y: np.ndarray, probs: np.ndarray,
                        min_count: int = 20) -> dict:
    """FPED/FNED across proxy groups on real (non-counterfactual) comments."""
    pred = probs.argmax(-1)
    fp_all = float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))
    fn_all = float(((pred == 0) & (y == 1)).sum() / max(1, (y == 1).sum()))
    rows = defaultdict(list)
    for i, t in enumerate(tags):
        for c, g in t.items():
            rows[(c, g)].append(i)
    cats = defaultdict(dict)
    for (c, g), ix in rows.items():
        ix = np.array(ix)
        if len(ix) < min_count:
            continue
        yy, pp = y[ix], pred[ix]
        neg, pos = (yy == 0).sum(), (yy == 1).sum()
        cats[c][g] = {"n": int(len(ix)),
                      "fpr": float(((pp == 1) & (yy == 0)).sum() / neg) if neg else None,
                      "fnr": float(((pp == 0) & (yy == 1)).sum() / pos) if pos else None,
                      "pred_pos_rate": float((pp == 1).mean())}
    out = {"overall_fpr": fp_all, "overall_fnr": fn_all, "by_category": {}}
    fpeds, fneds = [], []
    for c, groups in cats.items():
        if len(groups) < 2:
            continue
        fped = sum(abs(v["fpr"] - fp_all) for v in groups.values() if v["fpr"] is not None)
        fned = sum(abs(v["fnr"] - fn_all) for v in groups.values() if v["fnr"] is not None)
        dp = [v["pred_pos_rate"] for v in groups.values()]
        out["by_category"][c] = {"groups": groups, "fped": fped, "fned": fned, "dp_gap": max(dp) - min(dp)}
        fpeds.append(fped)
        fneds.append(fned)
    out["group_fped"] = float(np.mean(fpeds)) if fpeds else None
    out["group_fned"] = float(np.mean(fneds)) if fneds else None
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = [x[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def paired_bootstrap_recovery(gap_base: np.ndarray, gap_s: np.ndarray, gap_adv: np.ndarray,
                              n_boot: int = 1000, seed: int = 0) -> Optional[List[float]]:
    """CI of recovery = (B_base - B_s) / (B_base - B_adv), resampling pairs jointly."""
    rng = np.random.default_rng(seed)
    n = len(gap_base)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        ga = gap_base[ix].mean() - gap_adv[ix].mean()
        if abs(ga) < 1e-9:
            continue
        vals.append((gap_base[ix].mean() - gap_s[ix].mean()) / ga)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
