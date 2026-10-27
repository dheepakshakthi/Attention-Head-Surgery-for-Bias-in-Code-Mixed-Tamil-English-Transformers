"""Attention-head surgery: localise bias heads, select a minimal set by ablation,
re-weight their gates, and produce a report naming what was removed.

Data hygiene: localisation uses *train*-split counterfactual pairs, head-set
selection uses *dev* pairs, and all reported numbers use *test* pairs
(templates are split the same way), so the surgery is never tuned on test data.

Pipeline (every pass metered by :class:`CostTracker`, phase names in brackets):

1. [screen]   attribution-patching NIE estimate for all L x H heads,
2. [exact]    exact Vig NIE / NDE for the top-M screened heads,
3. [ablate]   zero-ablation of each candidate on dev: delta CF-gap and delta macro-F1,
4. [select]   evaluate prefix sets of the ranking; keep the smallest set reaching
              >= ``select_frac`` of the best bias reduction within the accuracy budget,
5. [gates]    re-weight: ``zero`` (g=0), ``scale`` (g=alpha) or ``learned`` - optimise only
              the k selected gates with CE + lambda * counterfactual-consistency.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .engine import CostTracker, encode, predict_proba
from .mediation import (Mediator, aggregate, category_weighted_score, proxy_attention_ratio,
                        run_attribution, run_exact, select_pairs)
from .metrics import cf_metrics, task_metrics
from .modeling import GateParam
from .utils import autocast_ctx, get_logger, make_grad_scaler

log = get_logger()
Head = Tuple[int, int]


def head_mask_for(L: int, H: int, heads: List[Head], value: float, device) -> torch.Tensor:
    m = torch.ones(L, H, device=device)
    for l, h in heads:
        m[l, h] = value
    return m


class DevProbe:
    """Cheap dev-set probe: CF gap on dev pairs + macro-F1 on a dev subsample."""

    def __init__(self, model, tok, pairs, texts, labels, cfg, tracker):
        self.model, self.tok, self.pairs = model, tok, pairs
        self.texts, self.labels = texts, np.asarray(labels)
        self.ml = cfg["train"]["max_length"]
        self.bs = cfg["train"].get("eval_batch_size", 64)
        self.fp16 = cfg["train"].get("fp16", True)
        self.tracker = tracker

    def __call__(self, head_mask: Optional[torch.Tensor] = None) -> dict:
        ta = [p["text_a"] for p in self.pairs] + [p["text_b"] for p in self.pairs] + list(self.texts)
        pr = predict_proba(self.model, self.tok, ta, self.ml, self.bs, head_mask=head_mask,
                           tracker=self.tracker, fp16=self.fp16)
        n = len(self.pairs)
        cf = cf_metrics(self.pairs, pr[:n], pr[n:2 * n])
        tm = task_metrics(self.labels, pr[2 * n:])
        return {"cf_gap": cf["cf_gap"], "cf_flip_rate": cf["cf_flip_rate"], "macro_f1": tm["macro_f1"],
                "by_category": cf["by_category"]}


# ---------------------------------------------------------------------------
def localize(model, tok, train_pairs: List[dict], cfg: dict, tracker: CostTracker) -> dict:
    scfg = cfg["surgery"]
    L, H = model.num_layers, model.num_heads
    cats = scfg.get("target_categories") or sorted({p["category"] for p in train_pairs})
    med = Mediator(model, tok, cfg["train"]["max_length"], cfg["train"].get("fp16", True), tracker)
    seed = cfg.get("seed", 42)

    with tracker.phase("screen"):
        sp = select_pairs(train_pairs, scfg.get("screen_pairs_per_category", 64), seed, cats)
        log.info(f"[surgery] screening {L * H} heads with attribution patching on {len(sp)} pairs")
        screen = aggregate(run_attribution(med, sp, "screen"), L, H)
    screen_score = category_weighted_score(screen, "nie", cats)

    M = min(scfg.get("exact_top_m", 24), L * H)
    order = np.argsort(-screen_score.ravel())[:M]
    cand = [(int(i // H), int(i % H)) for i in order]
    with tracker.phase("exact"):
        ep = select_pairs(train_pairs, scfg.get("exact_pairs_per_category", 24), seed + 1, cats)
        log.info(f"[surgery] exact NIE/NDE for {M} candidate heads on {len(ep)} pairs")
        recs = run_exact(med, ep, cand, scfg.get("mediator", "head_output"),
                         scfg.get("intervention_chunk", 48), want_nde=True, desc="exact")
        exact = aggregate(recs, L, H, heads=cand)
    exact_score = category_weighted_score(exact, "nie", cats)
    return {"categories": cats, "screen": screen, "screen_score": screen_score, "candidates": cand,
            "exact": exact, "exact_score": exact_score, "exact_records": recs}


def select_heads(model, probe: DevProbe, loc: dict, cfg: dict, tracker: CostTracker) -> dict:
    scfg = cfg["surgery"]
    L, H = model.num_layers, model.num_heads
    dev = next(model.parameters()).device
    cand = loc["candidates"]
    with tracker.phase("ablate"):
        base = probe(None)
        single = {}
        for (l, h) in tqdm(cand, desc="ablate", leave=False):
            r = probe(head_mask_for(L, H, [(l, h)], 0.0, dev))
            single[(l, h)] = {"d_cf_gap": r["cf_gap"] - base["cf_gap"], "d_macro_f1": r["macro_f1"] - base["macro_f1"]}
    # rank: exact oriented NIE (Vig) combined with measured ablation benefit
    ex = loc["exact_score"]
    w = scfg.get("ablation_weight", 0.5)
    nie_vals = np.array([ex[l, h] for l, h in cand])
    abl_vals = np.array([-single[c]["d_cf_gap"] for c in cand])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    combined = (1 - w) * z(nie_vals) + w * z(abl_vals)
    tol = scfg.get("max_f1_drop", 0.01)
    # discard heads whose solo ablation already breaks the task or increases bias
    ranked = [cand[i] for i in np.argsort(-combined)
              if single[cand[i]]["d_macro_f1"] >= -tol and abl_vals[i] > -1e-4]
    if not ranked:
        log.warning("[surgery] no candidate passes the ablation filters - using the top combined-score head")
        ranked = [cand[int(np.argmax(combined))]]
    k_max = min(scfg.get("k_max", 10), len(ranked))
    prefix = []
    with tracker.phase("select"):
        for k in range(1, k_max + 1):
            r = probe(head_mask_for(L, H, ranked[:k], 0.0, dev))
            prefix.append({"k": k, "heads": ranked[:k], "cf_gap": r["cf_gap"], "macro_f1": r["macro_f1"]})
    ok = [p for p in prefix if base["macro_f1"] - p["macro_f1"] <= tol]
    if not ok:
        log.warning("[surgery] no prefix within the accuracy budget - falling back to the single best head")
        chosen = ranked[:1]
    else:
        best_red = max(base["cf_gap"] - p["cf_gap"] for p in ok)
        frac = scfg.get("select_frac", 0.9)
        chosen = next(p["heads"] for p in ok if base["cf_gap"] - p["cf_gap"] >= frac * best_red)
    log.info(f"[surgery] selected {len(chosen)} heads: {[f'L{l}.H{h}' for l, h in chosen]}")
    return {"dev_base": base, "single_ablation": {f"L{l}.H{h}": v for (l, h), v in single.items()},
            "ranked": ranked, "prefix": prefix, "selected": chosen}


# ---------------------------------------------------------------------------
def learn_gates(model, tok, heads: List[Head], train_records: List[dict], train_pairs: List[dict],
                cfg: dict, tracker: CostTracker) -> torch.Tensor:
    """Optimise only the selected heads' gates: L = CE(task) + lambda * (y(a) - y(b))^2."""
    scfg = cfg["surgery"]
    gcfg = scfg.get("gate_training", {})
    L, H = model.num_layers, model.num_heads
    dev = next(model.parameters()).device
    fp16 = cfg["train"].get("fp16", True)
    ml = cfg["train"]["max_length"]
    mask = head_mask_for(L, H, heads, 0.0, dev)
    gp = GateParam(L, H, (mask == 0), init=gcfg.get("init", 0.5), max_gate=gcfg.get("max_gate", 1.0)).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()  # weights frozen; deterministic forward
    opt = torch.optim.Adam(gp.parameters(), lr=float(gcfg.get("lr", 0.05)))
    ce = nn.CrossEntropyLoss()
    rng = random.Random(cfg.get("seed", 42))
    bs, pbs = gcfg.get("batch_size", 16), gcfg.get("pair_batch_size", 16)
    lam = float(gcfg.get("cf_weight", 1.0))
    steps = int(gcfg.get("steps", 150))
    scaler = make_grad_scaler(dev, fp16)
    from .mediation import effect_measure
    with tracker.phase("gates"):
        for step in tqdm(range(steps), desc="gates", leave=False):
            tb = rng.sample(train_records, min(bs, len(train_records)))
            pb = rng.sample(train_pairs, min(pbs, len(train_pairs)))
            hm = gp()
            xb = encode(tok, [r["text"] for r in tb], ml, dev)
            ya = encode(tok, [p["text_a"] for p in pb], ml, dev)
            yb = encode(tok, [p["text_b"] for p in pb], ml, dev)
            for b in (xb, ya, yb):
                tracker.add_forward(b["attention_mask"], backward=True)
            with autocast_ctx(dev, fp16):
                lt = ce(model(**xb, head_mask=hm)["logits"].float(),
                        torch.tensor([r["label"] for r in tb], device=dev))
                ea = effect_measure(model(**ya, head_mask=hm)["logits"])
                eb = effect_measure(model(**yb, head_mask=hm)["logits"])
                lc = ((ea - eb) ** 2).mean()
                loss = lt + lam * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tracker.optimizer_steps += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return gp().detach()


def apply_surgery(model, gates: torch.Tensor):
    model.gates.copy_(gates.to(model.gates.device))


# ---------------------------------------------------------------------------
def name_removed(model, tok, loc: dict, sel: dict, gates: torch.Tensor, dev_pairs: List[dict],
                 cfg: dict, tracker: CostTracker) -> dict:
    """Human-readable account of what each removed / down-weighted head carried."""
    exact = loc["exact"]
    cats = [c for c in exact if c != "__all__"]
    heads = sel["selected"]
    with tracker.phase("report"):
        attn = proxy_attention_ratio(model, tok, [p for p in dev_pairs if p.get("span_a")][:200], heads,
                                     cfg["train"]["max_length"], cfg["train"].get("fp16", True), tracker)
    out = []
    recs = loc["exact_records"]
    cand_index = {tuple(h): i for i, h in enumerate(loc["candidates"])}
    for l, h in heads:
        name = f"L{l}.H{h}"
        per_cat = {c: {"nie_oriented": float(exact[c]["nie_oriented"][l, h]),
                       "nde_oriented": float(exact[c]["nde_oriented"][l, h]),
                       "nie_share_of_te": float(exact[c]["nie_share"][l, h])} for c in cats}
        top_cat = max(per_cat, key=lambda c: per_cat[c]["nie_share_of_te"])
        # groups whose swaps this head mediates most
        k = cand_index[(l, h)]
        gs = defaultdict(list)
        for r in recs:
            sgn = 1.0 if r["te"] >= 0 else -1.0
            gs[f"{r['category']}:{r['group_a']}->{r['group_b']}"].append(sgn * r["nie"][k])
        top_swaps = sorted(((g, float(np.mean(v)), len(v)) for g, v in gs.items() if len(v) >= 2),
                           key=lambda t: -t[1])[:5]
        ex = sorted(recs, key=lambda r: -abs(r["nie"][k]))[:3]
        out.append({
            "head": name, "gate": float(gates[l, h]),
            "primary_proxy_category": top_cat,
            "per_category": per_cat,
            "dev_single_ablation": sel["single_ablation"].get(name),
            "proxy_attention_ratio": attn.get(name),
            "top_mediated_swaps": [{"swap": g, "mean_oriented_nie": v, "n": n} for g, v, n in top_swaps],
            "examples": [{"text_a": r["text_a"], "text_b": r["text_b"], "te": r["te"], "nie": float(r["nie"][k])}
                         for r in ex],
        })
    return {"heads": out}


def run_surgery(model, tok, train_records, splits_pairs: Dict[str, List[dict]], dev_records, cfg: dict,
                tracker: CostTracker) -> dict:
    scfg = cfg["surgery"]
    rng = random.Random(cfg.get("seed", 42))
    train_pairs = splits_pairs["train"]
    dev_pairs = select_pairs(splits_pairs["dev"], scfg.get("dev_pairs_per_category", 60), cfg.get("seed", 42),
                             scfg.get("target_categories"))
    dev_sub = rng.sample(dev_records, min(len(dev_records), scfg.get("dev_task_examples", 400)))
    probe = DevProbe(model, tok, dev_pairs, [r["text"] for r in dev_sub], [r["label"] for r in dev_sub], cfg, tracker)

    loc = localize(model, tok, train_pairs, cfg, tracker)
    sel = select_heads(model, probe, loc, cfg, tracker)
    heads = sel["selected"]
    L, H = model.num_layers, model.num_heads
    dev = next(model.parameters()).device
    mode = scfg.get("reweight", "learned")
    if mode == "zero":
        gates = head_mask_for(L, H, heads, 0.0, dev)
    elif mode == "scale":
        gates = head_mask_for(L, H, heads, float(scfg.get("scale_alpha", 0.3)), dev)
    elif mode == "learned":
        gates = learn_gates(model, tok, heads, train_records, train_pairs, cfg, tracker)
    else:
        raise ValueError(mode)
    with tracker.phase("select"):
        after = probe(gates)
    # name what was removed *before* applying the gates, so proxy-attention ratios are
    # measured on the heads' original attention patterns
    report = name_removed(model, tok, loc, sel, gates, dev_pairs, cfg, tracker)
    apply_surgery(model, gates)
    report.update(reweight=mode, dev_before=sel["dev_base"], dev_after=after,
                  n_heads_total=L * H, n_heads_modified=len(heads), trainable_parameters=len(heads) if mode == "learned" else 0)
    return {"localization": loc, "selection": sel, "gates": gates.cpu(), "report": report}
