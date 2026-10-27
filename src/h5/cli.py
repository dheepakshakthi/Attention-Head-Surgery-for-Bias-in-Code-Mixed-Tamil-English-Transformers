"""Command-line entry point: ``python -m h5.cli <stage> --config configs/default.yaml``.

Stages (run in this order; ``all`` runs every stage):
  prepare      load + clean datasets, tag proxies, build counterfactual pairs / templates
  baseline     fine-tune the multilingual classifier (cost metered)
  analysis     full Vig et al. NIE/NDE for all L x H heads per proxy category (optional, analysis only)
  surgery      localise -> select -> re-weight heads on the baseline (cost metered)
  adversarial  full adversarial-debiasing fine-tune (cost metered)
  cda          counterfactual-data-augmentation fine-tune (optional extra baseline)
  evaluate     held-out evaluation of every trained method
  compare      recovery / cost ratio / verdict, REPORT.md and plots
  lexicon-export / lexicon-merge   native-speaker annotation round-trip for the proxy lexicon
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import counterfactual as cfm
from .config import load_config
from .data import label_names, prepare_datasets
from .engine import CostTracker, count_params, train_classifier
from .lexicon import ProxyLexicon, resolve_lexicon_path
from .metrics import task_metrics
from .modeling import HeadGatedClassifier
from .utils import ensure_dir, get_device, get_logger, load_json, load_jsonl, save_json, save_jsonl, set_seed

log = get_logger()


# ---------------------------------------------------------------------------
def paths(cfg):
    root = Path(cfg["output_dir"])
    return {"root": root, "data": root / "data", "cf": root / "cf"}


def load_frames(cfg):
    d = paths(cfg)["data"]
    if not (d / "train.jsonl").exists():
        raise FileNotFoundError("Run the 'prepare' stage first")
    return {p.stem: pd.DataFrame(load_jsonl(p)) for p in sorted(d.glob("*.jsonl"))}


def load_cf(cfg, split):
    d = paths(cfg)["cf"]
    pp, fp = d / f"{split}_pairs.jsonl", d / f"{split}_fills.jsonl"
    return {"pairs": load_jsonl(pp) if pp.exists() else [], "template_fills": load_jsonl(fp) if fp.exists() else []}


def load_attr(cfg, split):
    p = paths(cfg)["cf"] / f"attr_{split}.jsonl"
    return load_jsonl(p) if p.exists() else []


def get_lexicon(cfg):
    path = resolve_lexicon_path(cfg)
    log.info(f"[lexicon] using {path}")
    return ProxyLexicon.from_file(path, cfg.get("lexicon", {}).get("min_confidence", "low"))


def records(df):
    return [{"text": t, "label": int(y)} for t, y in zip(df["text"], df["label"])]


def num_labels(cfg):
    return len(label_names(cfg["data"].get("task", "offensive")))


def load_method_model(cfg, method, device):
    mdir = paths(cfg)["root"] / method / "model"
    meta = load_json(mdir / "meta.json")
    if meta.get("gates_only"):
        model, tok = HeadGatedClassifier.load(meta["base_model_dir"], device)
        model.gates.copy_(torch.load(mdir / "gates.pt", map_location="cpu").to(device))
        return model, tok
    return HeadGatedClassifier.load(mdir, device)


# ---------------------------------------------------------------------------
def stage_prepare(cfg):
    P = paths(cfg)
    frames = prepare_datasets(cfg)
    for k, v in frames.items():
        save_jsonl(v[["text", "label", "raw_label", "split", "source"]].to_dict("records"), P["data"] / f"{k}.jsonl")
    lex = get_lexicon(cfg)
    cf = cfm.build_all(cfg, frames, lex)
    cats = set(cfg["counterfactual"].get("categories", lex.all_category_names()))
    for split, d in cf.items():
        save_jsonl(d["pairs"], P["cf"] / f"{split}_pairs.jsonl")
        save_jsonl(d["template_fills"], P["cf"] / f"{split}_fills.jsonl")
        save_jsonl(cfm.attribute_records(lex, frames[split], d["pairs"], cats), P["cf"] / f"attr_{split}.jsonl")
    cov = cfm.proxy_coverage(lex, frames["train"])
    cov.to_csv(P["cf"] / "lexicon_coverage_train.csv", index=False)
    log.info(f"[prepare] lexicon coverage on train:\n{cov.to_string(index=False)}")


def stage_baseline(cfg, device):
    frames = load_frames(cfg)
    out = ensure_dir(paths(cfg)["root"] / "baseline")
    model = HeadGatedClassifier.from_pretrained_encoder(cfg["model"]["name"], num_labels(cfg),
                                                       cfg["model"].get("dropout", 0.1)).to(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tracker = CostTracker(count_params(model.encoder), "baseline")
    hist = train_classifier(model, tok, records(frames["train"]), records(frames["dev"]), cfg["train"], tracker,
                            task_metrics, seed=cfg.get("seed", 42), phase="train")
    model.save(out / "model", tok)
    save_json(hist, out / "train_history.json")
    save_json(tracker.to_dict(), out / "cost.json")


def stage_analysis(cfg, device):
    """Full-resolution Vig et al. mediation over all heads on *test-free* dev pairs (analysis only)."""
    from .mediation import Mediator, aggregate, run_exact, select_pairs
    acfg = cfg["analysis"]
    out = ensure_dir(paths(cfg)["root"] / "analysis")
    model, tok = HeadGatedClassifier.load(paths(cfg)["root"] / "baseline" / "model", device)
    tracker = CostTracker(count_params(model.encoder), "analysis")
    pairs = select_pairs(load_cf(cfg, "dev")["pairs"], acfg.get("pairs_per_category", 40), cfg.get("seed", 42),
                         acfg.get("categories"))
    heads = [(l, h) for l in range(model.num_layers) for h in range(model.num_heads)]
    med = Mediator(model, tok, cfg["train"]["max_length"], cfg["train"].get("fp16", True), tracker)
    tables = {}
    summary = {}
    for mediator in acfg.get("mediators", ["head_output"]):
        with tracker.phase(f"exact_{mediator}"):
            recs = run_exact(med, pairs, heads, mediator, acfg.get("intervention_chunk", 48),
                             want_nde=True, desc=f"analysis[{mediator}]")
        agg = aggregate(recs, model.num_layers, model.num_heads, heads=heads)
        for c, e in agg.items():
            for k in ("nie_oriented", "nde_oriented", "nie_share", "nie_abs"):
                tables[f"{mediator}:{c}__{k}"] = e[k]
            top = np.dstack(np.unravel_index(np.argsort(-e["nie_oriented"].ravel())[:10], e["nie_oriented"].shape))[0]
            summary.setdefault(mediator, {})[c] = {
                "n_pairs": e["n"], "mean_abs_te": e["mean_abs_te"],
                "sum_nie_share_top10": float(sum(e["nie_share"][l, h] for l, h in top)),
                "top10_heads": [{"head": f"L{l}.H{h}", "nie_oriented": float(e["nie_oriented"][l, h]),
                                 "nde_oriented": float(e["nde_oriented"][l, h]),
                                 "nie_share": float(e["nie_share"][l, h])} for l, h in top]}
    # agreement between mediator types (Spearman over heads)
    if len(acfg.get("mediators", [])) == 2:
        from scipy.stats import spearmanr
        m1, m2 = acfg["mediators"]
        a, b = tables[f"{m1}:__all____nie_oriented"], tables[f"{m2}:__all____nie_oriented"]
        summary["mediator_rank_agreement_spearman"] = float(spearmanr(a.ravel(), b.ravel()).correlation)
    np.savez(out / "mediation_tables.npz", **{k.replace(":", "__"): v for k, v in tables.items()})
    save_json(summary, out / "mediation_summary.json")
    save_json(tracker.to_dict(), out / "cost.json")


def stage_surgery(cfg, device):
    from .surgery import run_surgery
    frames = load_frames(cfg)
    out = ensure_dir(paths(cfg)["root"] / "surgery")
    base_dir = paths(cfg)["root"] / "baseline" / "model"
    model, tok = HeadGatedClassifier.load(base_dir, device)
    tracker = CostTracker(count_params(model.encoder), "surgery")
    splits = {"train": load_cf(cfg, "train")["pairs"], "dev": load_cf(cfg, "dev")["pairs"]}
    res = run_surgery(model, tok, records(frames["train"]), splits, records(frames["dev"]), cfg, tracker)
    mdir = ensure_dir(out / "model")
    torch.save(res["gates"], mdir / "gates.pt")
    save_json({"gates_only": True, "base_model_dir": str(base_dir)}, mdir / "meta.json")
    save_json(res["report"], out / "surgery_report.json")
    sel = res["selection"]
    save_json({"selected": [f"L{l}.H{h}" for l, h in sel["selected"]],
               "ranked": [f"L{l}.H{h}" for l, h in sel["ranked"]],
               "prefix": [dict(p, heads=[f"L{l}.H{h}" for l, h in p["heads"]]) for p in sel["prefix"]],
               "single_ablation": sel["single_ablation"], "dev_base": sel["dev_base"]}, out / "selection.json")
    loc = res["localization"]
    tabs = {}
    for name in ("screen", "exact"):
        for c, e in loc[name].items():
            for k in ("nie_oriented", "nde_oriented", "nie_share"):
                if k in e:
                    tabs[f"{name}_{c}__{k}"] = e[k]
    np.savez(out / "screen_tables.npz", **tabs)
    save_json(tracker.to_dict(), out / "cost.json")


def stage_adversarial(cfg, device):
    from .adversarial import train_adversarial
    frames = load_frames(cfg)
    out = ensure_dir(paths(cfg)["root"] / "adversarial")
    tracker = CostTracker(0, "adversarial")
    model, tok, hist = train_adversarial(cfg, records(frames["train"]), records(frames["dev"]), load_attr(cfg, "train"),
                                         num_labels(cfg), device, tracker, task_metrics)
    tracker.n_params = count_params(model.encoder)
    model.save(out / "model", tok)
    save_json(hist, out / "train_history.json")
    save_json(tracker.to_dict(), out / "cost.json")


def stage_cda(cfg, device):
    """Counterfactual data augmentation: train on originals + label-preserving swaps."""
    frames = load_frames(cfg)
    out = ensure_dir(paths(cfg)["root"] / "cda")
    pairs = [p for p in load_cf(cfg, "train")["pairs"] if p["source"] == "natural"]
    aug = records(frames["train"]) + [{"text": p["text_b"], "label": p["label"]} for p in pairs]
    model = HeadGatedClassifier.from_pretrained_encoder(cfg["model"]["name"], num_labels(cfg),
                                                       cfg["model"].get("dropout", 0.1)).to(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tracker = CostTracker(count_params(model.encoder), "cda")
    hist = train_classifier(model, tok, aug, records(frames["dev"]), cfg["train"], tracker, task_metrics,
                            seed=cfg.get("seed", 42), phase="cda_train")
    model.save(out / "model", tok)
    save_json(hist, out / "train_history.json")
    save_json(tracker.to_dict(), out / "cost.json")


def stage_evaluate(cfg, device, methods=None):
    from .evaluate import evaluate_model
    frames = load_frames(cfg)
    lex = get_lexicon(cfg)
    eval_splits = [s for s in frames if s not in ("train", "dev")]
    cf = {s: load_cf(cfg, s) for s in eval_splits}
    attr = load_attr(cfg, "test")
    for m in methods or ["baseline", "surgery", "adversarial", "cda"]:
        if not (paths(cfg)["root"] / m / "model" / "meta.json").exists():
            log.info(f"[evaluate] skip {m} (no model)")
            continue
        model, tok = load_method_model(cfg, m, device)
        evaluate_model(model, tok, cfg, frames, cf, lex, attr, paths(cfg)["root"] / m)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_compare(cfg):
    from .compare import compare
    compare(cfg)


# --- lexicon annotation round-trip ---------------------------------------
def stage_lexicon_export(cfg):
    """Write a CSV that native speakers fill in (keep / drop / fix, dialect + region notes)."""
    lex_spec = load_json(resolve_lexicon_path(cfg))
    frames = load_frames(cfg) if (paths(cfg)["data"] / "train.jsonl").exists() else None
    lex = ProxyLexicon(lex_spec)
    counts = {}
    if frames is not None:
        for t in frames["train"]["text"]:
            for m in lex.find(t):
                counts[m.form.lower()] = counts.get(m.form.lower(), 0) + 1
    out = ensure_dir(paths(cfg)["root"] / "lexicon") / "annotation_sheet.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "row_id", "slot", "script", "group", "form", "train_count",
                    "keep(y/n)", "corrected_form", "is_offensive_or_slur(y/n)", "annotator_notes"])
        for c, cat in lex_spec["categories"].items():
            for r in cat["rows"]:
                for g, forms in r["forms"].items():
                    for form in forms:
                        w.writerow([c, r["id"], r["slot"], r["script"], g, form, counts.get(form.lower(), 0), "", "", "", ""])
    log.info(f"[lexicon] annotation sheet -> {out}")


def stage_lexicon_merge(cfg, sheets):
    """Merge one or more filled sheets: a form is kept if a strict majority of annotators keep it
    and nobody flags it as a slur. Writes resources/proxy_lexicon.validated.json."""
    spec = load_json(cfg["lexicon"].get("seed_path", "resources/proxy_lexicon.seed.json"))
    votes = {}
    for s in sheets:
        for row in csv.DictReader(open(s, encoding="utf-8")):
            key = (row["category"], row["row_id"], row["group"], row["form"])
            v = votes.setdefault(key, {"keep": 0, "drop": 0, "slur": False, "fix": []})
            k = row.get("keep(y/n)", "").strip().lower()
            v["keep" if k == "y" else "drop"] += 1 if k in ("y", "n") else 0
            v["slur"] |= row.get("is_offensive_or_slur(y/n)", "").strip().lower() == "y"
            if row.get("corrected_form", "").strip():
                v["fix"].append(row["corrected_form"].strip())
    for c, cat in spec["categories"].items():
        for r in cat["rows"]:
            for g in list(r["forms"]):
                kept = []
                for form in r["forms"][g]:
                    v = votes.get((c, r["id"], g, form))
                    if v is None or v["slur"] or v["keep"] <= v["drop"]:
                        continue
                    kept.append(max(set(v["fix"]), key=v["fix"].count) if v["fix"] else form)
                r["forms"][g] = kept
            r["confidence"] = "high"
    spec["version"] = spec.get("version", "seed") + "+validated"
    spec["status"] = f"Validated by majority vote over {len(sheets)} annotation sheet(s)."
    dst = Path(cfg["lexicon"].get("validated_path", "resources/proxy_lexicon.validated.json"))
    save_json(spec, dst)
    log.info(f"[lexicon] validated lexicon -> {dst}")


# ---------------------------------------------------------------------------
STAGES = ["prepare", "baseline", "analysis", "surgery", "adversarial", "cda", "evaluate", "compare"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h5", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all", "lexicon-export", "lexicon-merge"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. train.epochs=2")
    ap.add_argument("--sheets", nargs="*", default=[], help="filled annotation CSVs (lexicon-merge)")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods for 'evaluate'")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    root = ensure_dir(cfg["output_dir"])
    get_logger(log_file=root / "run.log")
    save_json(cfg, root / f"config_{args.stage}.json")
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    log.info(f"[h5] stage={args.stage} device={device} model={cfg['model']['name']}")

    todo = STAGES if args.stage == "all" else [args.stage]
    for st in todo:
        if st == "analysis" and not cfg.get("analysis", {}).get("enabled", True):
            log.info("analysis disabled (analysis.enabled=false) - skipping")
            continue
        if st == "cda" and not cfg.get("cda", {}).get("enabled", False):
            log.info("CDA baseline disabled - enable with --set cda.enabled=true")
            continue
        set_seed(cfg.get("seed", 42))
        log.info(f"===== stage: {st} =====")
        if st == "prepare":
            stage_prepare(cfg)
        elif st == "baseline":
            stage_baseline(cfg, device)
        elif st == "analysis":
            stage_analysis(cfg, device)
        elif st == "surgery":
            stage_surgery(cfg, device)
        elif st == "adversarial":
            stage_adversarial(cfg, device)
        elif st == "cda":
            stage_cda(cfg, device)
        elif st == "evaluate":
            stage_evaluate(cfg, device, args.methods)
        elif st == "compare":
            stage_compare(cfg)
        elif st == "lexicon-export":
            stage_lexicon_export(cfg)
        elif st == "lexicon-merge":
            if not args.sheets:
                sys.exit("--sheets is required for lexicon-merge")
            stage_lexicon_merge(cfg, args.sheets)


if __name__ == "__main__":
    main()
