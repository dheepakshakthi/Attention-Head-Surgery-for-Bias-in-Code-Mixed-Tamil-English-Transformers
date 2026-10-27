"""Held-out evaluation of one model: task quality + every fairness metric."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .adversarial import build_attribute_pools, probe_leakage
from .counterfactual import tag_text
from .engine import predict_proba
from .metrics import cf_metrics, group_error_metrics, task_metrics, template_metrics
from .utils import ensure_dir, get_logger, save_json

log = get_logger()


def evaluate_model(model, tok, cfg: dict, frames: Dict[str, "pd.DataFrame"], cf: dict, lex,
                   attr_records: List[dict], out_dir) -> dict:
    out_dir = ensure_dir(out_dir)
    ml = cfg["train"]["max_length"]
    bs = cfg["train"].get("eval_batch_size", 64)
    fp16 = cfg["train"].get("fp16", True)
    ecfg = cfg.get("evaluation", {})
    n_boot = ecfg.get("n_bootstrap", 1000)
    res = {"task": {}, "cf": {}, "templates": {}, "groups": {}}
    eval_splits = [s for s in frames if s not in ("train", "dev")]
    for split in eval_splits:
        df = frames[split]
        probs = predict_proba(model, tok, df["text"].tolist(), ml, bs, fp16=fp16, desc=f"eval {split}")
        y = df["label"].to_numpy()
        res["task"][split] = task_metrics(y, probs)
        tags = [tag_text(lex, t) for t in df["text"]]
        res["groups"][split] = group_error_metrics(tags, y, probs, ecfg.get("min_group_count", 20))
        pairs = cf.get(split, {}).get("pairs", [])
        if pairs:
            pa = predict_proba(model, tok, [p["text_a"] for p in pairs], ml, bs, fp16=fp16)
            pb = predict_proba(model, tok, [p["text_b"] for p in pairs], ml, bs, fp16=fp16)
            res["cf"][split] = cf_metrics(pairs, pa, pb, n_boot=n_boot, seed=cfg.get("seed", 42))
            # per-pair gaps are saved for paired bootstrap in the comparison stage
            np.save(Path(out_dir) / f"cf_gaps_{split}.npy", np.abs(pa[:, 1] - pb[:, 1]))
            nat = [i for i, p in enumerate(pairs) if p["source"] == "natural"]
            tpl = [i for i, p in enumerate(pairs) if p["source"] == "template"]
            for name, ix in (("natural", nat), ("template", tpl)):
                if ix:
                    res["cf"][split][f"{name}_only"] = cf_metrics([pairs[i] for i in ix], pa[ix], pb[ix])
        fills = cf.get(split, {}).get("template_fills", [])
        if fills:
            pf = predict_proba(model, tok, [f["text"] for f in fills], ml, bs, fp16=fp16)
            res["templates"][split] = template_metrics(fills, pf)
        log.info(f"[eval] {split}: " + ", ".join(f"{k}={v:.4f}" for k, v in res["task"][split].items())
                 + (f", cf_gap={res['cf'][split]['cf_gap']:.4f}" if split in res["cf"] else ""))
    if ecfg.get("probe_leakage", True) and attr_records:
        pools = build_attribute_pools(attr_records, 10)
        res["leakage_probe"] = probe_leakage(model, tok, pools, cfg["train"], next(model.parameters()).device,
                                             cfg.get("seed", 42))
    save_json(res, Path(out_dir) / "eval.json")
    return res
