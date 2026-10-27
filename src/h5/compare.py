"""Compare baseline / head surgery / adversarial (/ CDA) and test the H5 claim:

  "head surgery recovers most of the adversarial fairness gain at ~5 % of its
   training cost, and names what was removed".
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .metrics import paired_bootstrap_recovery
from .utils import get_logger, load_json, save_json

log = get_logger()

BIAS_METRICS = [  # (path into eval.json, label); all are lower-is-better
    (("cf", "test", "cf_gap"), "CF gap (all pairs)"),
    (("cf", "test", "natural_only", "cf_gap"), "CF gap (natural pairs)"),
    (("cf", "test", "template_only", "cf_gap"), "CF gap (template pairs)"),
    (("cf", "test", "cf_flip_rate"), "CF flip rate"),
    (("templates", "test", "template_fped"), "Template FPED"),
    (("templates", "test", "template_range"), "Template range"),
    (("groups", "test", "group_fped"), "Group FPED (real)"),
    (("groups", "test", "group_fned"), "Group FNED (real)"),
]


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _load(out_root: Path, name: str) -> Optional[dict]:
    p = out_root / name / "eval.json"
    return load_json(p) if p.exists() else None


def _cost(out_root: Path, name: str) -> Optional[dict]:
    p = out_root / name / "cost.json"
    return load_json(p) if p.exists() else None


def compare(cfg: dict) -> dict:
    out_root = Path(cfg["output_dir"])
    ccfg = cfg.get("comparison", {})
    methods = ["baseline", "surgery", "adversarial", "cda"]
    evals = {m: _load(out_root, m) for m in methods}
    costs = {m: _cost(out_root, m) for m in methods}
    if not (evals["baseline"] and evals["surgery"] and evals["adversarial"]):
        raise RuntimeError("compare needs outputs/{baseline,surgery,adversarial}/eval.json")

    table = {}
    for path, label in BIAS_METRICS:
        row = {m: _get(evals[m], path) for m in methods if evals[m]}
        b, s, a = row.get("baseline"), row.get("surgery"), row.get("adversarial")
        if None not in (b, s, a):
            ga, gs = b - a, b - s
            row["gain_adv"], row["gain_surgery"] = ga, gs
            row["recovery"] = gs / ga if abs(ga) > 1e-9 else None
        table[label] = row
    task = {}
    for split in evals["baseline"]["task"]:
        task[split] = {m: evals[m]["task"].get(split) for m in methods if evals[m]}

    # paired bootstrap CI on the primary metric
    gaps = {m: out_root / m / "cf_gaps_test.npy" for m in ("baseline", "surgery", "adversarial")}
    ci = None
    if all(p.exists() for p in gaps.values()):
        g = {m: np.load(p) for m, p in gaps.items()}
        if len({len(v) for v in g.values()}) == 1:
            ci = paired_bootstrap_recovery(g["baseline"], g["surgery"], g["adversarial"],
                                           cfg.get("evaluation", {}).get("n_bootstrap", 1000), cfg.get("seed", 42))

    # cost
    cs, ca, cb = costs["surgery"], costs["adversarial"], costs["baseline"]
    cost = {}
    if cs and ca:
        cost = {
            "surgery_flops": cs["approx_flops"], "adversarial_flops": ca["approx_flops"],
            "surgery_wall_s": cs["wall_seconds"], "adversarial_wall_s": ca["wall_seconds"],
            "ratio_flops": cs["approx_flops"] / ca["approx_flops"],
            "ratio_wall": cs["wall_seconds"] / max(1e-9, ca["wall_seconds"]),
            "surgery_phases": cs.get("phases", {}),
        }
        an = _cost(out_root, "analysis")
        if an:  # optional full L x H Vig analysis, reported separately
            cost["ratio_flops_incl_full_analysis"] = (cs["approx_flops"] + an["approx_flops"]) / ca["approx_flops"]
            cost["ratio_wall_incl_full_analysis"] = (cs["wall_seconds"] + an["wall_seconds"]) / max(1e-9, ca["wall_seconds"])
        if cb:
            cost["baseline_flops"] = cb["approx_flops"]
            cost["surgery_vs_baseline_flops"] = cs["approx_flops"] / cb["approx_flops"]

    primary = table["CF gap (all pairs)"]
    rec = primary.get("recovery")
    rec_thr = ccfg.get("recovery_threshold", 0.8)
    cost_target = ccfg.get("cost_target", 0.05)
    cost_tol = ccfg.get("cost_tolerance", 2.0)
    f1_tol = ccfg.get("max_f1_drop", 0.02)
    f1_b = task["test"]["baseline"]["macro_f1"]
    f1_s = task["test"]["surgery"]["macro_f1"]
    f1_a = task["test"]["adversarial"]["macro_f1"]
    srep = Path(out_root / "surgery" / "surgery_report.json")
    named = load_json(srep)["heads"] if srep.exists() else []
    verdict = {
        "recovery": rec, "recovery_ci95": ci, "recovery_threshold": rec_thr,
        "recovers_most": bool(rec is not None and rec >= rec_thr),
        "adversarial_gain_positive": bool((primary.get("gain_adv") or 0) > 0),
        "cost_ratio_flops": cost.get("ratio_flops"), "cost_target": cost_target,
        "cost_about_5pct": bool(cost.get("ratio_flops") is not None and cost["ratio_flops"] <= cost_target * cost_tol),
        "surgery_f1_drop": f1_b - f1_s, "adversarial_f1_drop": f1_b - f1_a,
        "surgery_task_preserved": bool(f1_b - f1_s <= f1_tol),
        "names_what_was_removed": bool(named),
    }
    verdict["claim_supported"] = all(verdict[k] for k in ("recovers_most", "adversarial_gain_positive",
                                                          "cost_about_5pct", "surgery_task_preserved",
                                                          "names_what_was_removed"))
    result = {"bias_table": table, "task": task, "cost": cost, "verdict": verdict,
              "removed": [{"head": h["head"], "gate": h["gate"], "primary_proxy_category": h["primary_proxy_category"]}
                          for h in named],
              "leakage_probe": {m: evals[m].get("leakage_probe") for m in methods if evals[m]}}
    save_json(result, out_root / "comparison.json")
    (out_root / "REPORT.md").write_text(render_markdown(result), encoding="utf-8")
    try:
        make_plots(cfg, result)
    except Exception as e:  # plotting must never break the pipeline
        log.warning(f"plotting failed: {e}")
    log.info(f"[compare] recovery={rec}, cost_ratio={cost.get('ratio_flops')}, claim_supported={verdict['claim_supported']}")
    return result


def _fmt(v, pct=False):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, pct) for x in v) + "]"
    return f"{100 * v:.1f}%" if pct else f"{v:.4f}"


def render_markdown(r: dict) -> str:
    v, c = r["verdict"], r["cost"]
    L = ["# H5 results: head surgery vs adversarial debiasing", ""]
    L += ["## Verdict", "",
          f"- Fairness recovery (primary: CF gap): **{_fmt(v['recovery'], True)}** "
          f"(95% CI {_fmt(v['recovery_ci95'], True)}; threshold {_fmt(v['recovery_threshold'], True)}) -> "
          f"{'PASS' if v['recovers_most'] else 'FAIL'}",
          f"- Cost ratio (FLOPs, surgery / adversarial): **{_fmt(v['cost_ratio_flops'], True)}** "
          f"(target ~{_fmt(v['cost_target'], True)}) -> {'PASS' if v['cost_about_5pct'] else 'FAIL'}",
          f"- Wall-clock ratio: {_fmt(c.get('ratio_wall'), True)}; incl. full L x H analysis: "
          f"{_fmt(c.get('ratio_flops_incl_full_analysis'), True)}",
          f"- Macro-F1 drop: surgery {_fmt(v['surgery_f1_drop'])}, adversarial {_fmt(v['adversarial_f1_drop'])} -> "
          f"{'PASS' if v['surgery_task_preserved'] else 'FAIL'}",
          f"- Names what was removed: {'yes' if v['names_what_was_removed'] else 'no'}",
          f"- **Claim supported: {v['claim_supported']}**", ""]
    L += ["## Bias metrics (test, lower is better)", "",
          "| metric | baseline | surgery | adversarial | cda | recovery |", "|---|---|---|---|---|---|"]
    for k, row in r["bias_table"].items():
        L.append(f"| {k} | {_fmt(row.get('baseline'))} | {_fmt(row.get('surgery'))} | {_fmt(row.get('adversarial'))} "
                 f"| {_fmt(row.get('cda'))} | {_fmt(row.get('recovery'), True)} |")
    L += ["", "## Task metrics", "", "| split | method | accuracy | macro-F1 |", "|---|---|---|---|"]
    for split, ms in r["task"].items():
        for m, t in ms.items():
            if t:
                L.append(f"| {split} | {m} | {_fmt(t['accuracy'])} | {_fmt(t['macro_f1'])} |")
    L += ["", "## What the surgery removed", "", "| head | gate | primary proxy category |", "|---|---|---|"]
    for h in r["removed"]:
        L.append(f"| {h['head']} | {h['gate']:.3f} | {h['primary_proxy_category']} |")
    L += ["", "Full per-head details (per-category NIE/NDE, proxy-attention ratio, top mediated swaps, examples): "
          "`outputs/surgery/surgery_report.json`.", ""]
    L += ["## Proxy leakage probe (linear probe accuracy on [CLS]; lower = less proxy information)", ""]
    for m, p in r["leakage_probe"].items():
        if p:
            L.append(f"- {m}: " + ", ".join(f"{c}={_fmt(x['probe_accuracy'])} (chance {_fmt(x['chance'])})" for c, x in p.items()))
    return "\n".join(L) + "\n"


def make_plots(cfg: dict, r: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(cfg["output_dir"]) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    # 1) bias bars
    rows = [(k, v) for k, v in r["bias_table"].items() if v.get("baseline") is not None]
    ms = ["baseline", "surgery", "adversarial", "cda"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    for i, m in enumerate(ms):
        vals = [v.get(m) if v.get(m) is not None else 0 for _, v in rows]
        ax.bar(np.arange(len(rows)) + i * w, vals, w, label=m)
    ax.set_xticks(np.arange(len(rows)) + 1.5 * w)
    ax.set_xticklabels([k for k, _ in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("lower is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bias_metrics.png", dpi=150)
    plt.close(fig)
    # 2) cost
    c = r["cost"]
    if c:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["surgery", "adversarial"], [c["surgery_flops"], c["adversarial_flops"]], color=["tab:green", "tab:red"])
        ax.set_yscale("log")
        ax.set_ylabel("approx. FLOPs")
        ax.set_title(f"cost ratio = {100 * c['ratio_flops']:.1f}%")
        fig.tight_layout()
        fig.savefig(out / "cost.png", dpi=150)
        plt.close(fig)
    # 3) mediation heatmaps (full analysis if present, else the surgery screen)
    for name in ("analysis/mediation_tables.npz", "surgery/screen_tables.npz"):
        p = Path(cfg["output_dir"]) / name
        if p.exists():
            plot_heatmaps(p, out, prefix=name.split("/")[0])


def plot_heatmaps(npz_path: Path, out: Path, prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    keys = [k for k in d.files if k.endswith("nie_oriented")]
    if not keys:
        return
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        m = d[k]
        lim = np.nanmax(np.abs(m)) or 1.0
        im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(k.replace("__nie_oriented", ""), fontsize=8)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{prefix}: oriented NIE per head (Vig et al.)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_nie_heatmaps.png", dpi=150)
    plt.close(fig)
