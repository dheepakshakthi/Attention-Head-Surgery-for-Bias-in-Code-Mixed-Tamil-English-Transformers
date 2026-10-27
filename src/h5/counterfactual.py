"""Counterfactual augmentation protocol for Tamil-English code-mixing.

Produces three artefacts per split:

1. **natural pairs** - real comments containing a lexicon proxy, paired with a
   copy in which only the proxy has been swapped to another group of the same
   lexicon row (same slot, same script, suffix preserved). Label is preserved.
2. **rule-based pairs** - script switch (Tamil script -> Roman transliteration)
   and transliteration-style switch (long phonetic -> short casual spelling).
3. **template pairs** - neutral templates (gold label 0) filled with every
   group, in the spirit of Vig et al. (2020).

Protocol rules (documented in README):
* at most ``max_swaps_per_text`` target groups are sampled per matched proxy;
* only the *first* proxy of a category in a text is swapped (single
  intervention => causal attribution to one token);
* script/style pairs are skipped when the transformation is the identity;
* templates are split train/dev/test round-robin within (slot, script).
"""
from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd

from .lexicon import ProxyLexicon
from .transliterate import has_long_style, has_tamil_script, tamil_fraction, to_short_style, transliterate_tamil
from .utils import get_logger, load_json

log = get_logger()


def script_group(text: str) -> str:
    return "tamil_script" if tamil_fraction(text) >= 0.5 else "roman"


def style_group(text: str) -> str:
    if has_tamil_script(text):
        return "n/a"
    return "long_style" if has_long_style(text) else "short_style"


def tag_text(lex: ProxyLexicon, text: str) -> Dict[str, str]:
    """category -> group for the first proxy of each lexicon category, plus rule categories."""
    tags = {}
    for m in lex.find(text):
        tags.setdefault(m.category, m.group)
    tags["script"] = script_group(text)
    sg = style_group(text)
    if sg != "n/a":
        tags["translit_style"] = sg
    return tags


def natural_pairs(lex: ProxyLexicon, df: pd.DataFrame, split: str, cfg: dict, rng: random.Random) -> List[dict]:
    ccfg = cfg["counterfactual"]
    k = ccfg.get("max_swaps_per_text", 2)
    cats = set(ccfg.get("categories", lex.all_category_names()))
    pairs = []
    for idx, (text, label) in enumerate(zip(df["text"], df["label"])):
        seen = set()
        for m in lex.find(text):
            if m.category not in cats or m.category in seen:
                continue
            seen.add(m.category)
            row = lex.row(m.category, m.row_id)
            targets = [g for g in row.forms if g != m.group]
            rng.shuffle(targets)
            for g in targets[:k]:
                tb = lex.substitute(text, m, g)
                if tb and tb != text:
                    pairs.append(dict(split=split, source="natural", category=m.category, slot=m.slot,
                                      group_a=m.group, group_b=g, text_a=text, text_b=tb, label=int(label),
                                      span_a=[m.start, m.end], orig_index=idx))
        if "script" in cats and has_tamil_script(text) and tamil_fraction(text) >= 0.5:
            tb = transliterate_tamil(text)
            if tb != text:
                pairs.append(dict(split=split, source="natural", category="script", slot="script",
                                  group_a="tamil_script", group_b="roman", text_a=text, text_b=tb,
                                  label=int(label), span_a=None, orig_index=idx))
        if "translit_style" in cats and has_long_style(text):
            tb = to_short_style(text)
            if tb != text:
                pairs.append(dict(split=split, source="natural", category="translit_style", slot="style",
                                  group_a="long_style", group_b="short_style", text_a=text, text_b=tb,
                                  label=int(label), span_a=None, orig_index=idx))
    return pairs


def _template_split(i: int, n: int) -> str:
    if n == 1:
        return "test"
    return ["test", "dev", "train"][i % 3]


def template_pairs(lex: ProxyLexicon, templates: List[dict], cats) -> Dict[str, List[dict]]:
    by_key = defaultdict(list)
    for t in templates:
        by_key[(t["slot"], t["script"])].append(t)
    out = defaultdict(list)
    tid = 0
    for (slot, script), ts in by_key.items():
        rows = [(c, r) for c, r in lex.rows_for_slot(slot, script) if c in cats]
        for i, t in enumerate(ts):
            split = _template_split(i, len(ts))
            for cname, row in rows:
                fills = {g: row.canonical(g) for g in row.forms}
                x0 = t["text"].index("{X}")
                for ga, gb in itertools.combinations(sorted(fills), 2):
                    ta = t["text"].replace("{X}", fills[ga])
                    tb = t["text"].replace("{X}", fills[gb])
                    out[split].append(dict(split=split, source="template", category=cname, slot=slot,
                                           group_a=ga, group_b=gb, text_a=ta, text_b=tb, label=0,
                                           span_a=[x0, x0 + len(fills[ga])], template_id=f"t{tid}",
                                           row_id=row.id))
            tid += 1
    return out


def template_fills(lex: ProxyLexicon, templates: List[dict], split: str, cats) -> List[dict]:
    """Every (template, group) fill - used for the template disparity metric."""
    by_key = defaultdict(list)
    for t in templates:
        by_key[(t["slot"], t["script"])].append(t)
    out, tid = [], 0
    for (slot, script), ts in by_key.items():
        rows = [(c, r) for c, r in lex.rows_for_slot(slot, script) if c in cats]
        for i, t in enumerate(ts):
            s = _template_split(i, len(ts))
            for cname, row in rows:
                if s == split:
                    for g in row.forms:
                        out.append(dict(template_id=f"t{tid}", row_id=row.id, category=cname, group=g,
                                        text=t["text"].replace("{X}", row.canonical(g)), label=0))
            tid += 1
    return out


def cap_pairs(pairs: List[dict], cap: int, rng: random.Random) -> List[dict]:
    """Cap the number of pairs per category (keeps class balance of rare categories)."""
    if not cap:
        return pairs
    by = defaultdict(list)
    for p in pairs:
        by[p["category"]].append(p)
    out = []
    for c, ps in by.items():
        if len(ps) > cap:
            ps = rng.sample(ps, cap)
        out.extend(ps)
    return out


def attribute_records(lex: ProxyLexicon, df: pd.DataFrame, pairs: List[dict], cats) -> List[dict]:
    """(text, category, group) records for the adversary of adversarial debiasing."""
    recs = []
    for text in df["text"]:
        for c, g in tag_text(lex, text).items():
            if c in cats:
                recs.append(dict(text=text, category=c, group=g))
    for p in pairs:
        recs.append(dict(text=p["text_b"], category=p["category"], group=p["group_b"]))
    return recs


def build_all(cfg: dict, frames: Dict[str, pd.DataFrame], lex: ProxyLexicon) -> dict:
    ccfg = cfg["counterfactual"]
    rng = random.Random(cfg.get("seed", 42))
    cats = set(ccfg.get("categories", lex.all_category_names()))
    templates = load_json(ccfg.get("templates_path", "resources/templates.json"))["templates"]
    tpairs = template_pairs(lex, templates, cats)
    result = {}
    for split, df in frames.items():
        tsplit = split if split in ("train", "dev") else "test"
        nat = natural_pairs(lex, df, split, cfg, rng)
        cap = ccfg.get("max_pairs_per_category_train" if split == "train" else "max_pairs_per_category_eval")
        nat = cap_pairs(nat, cap, rng)
        tp = [dict(p, split=split) for p in tpairs.get(tsplit, [])] if ccfg.get("use_templates", True) else []
        fills = template_fills(lex, templates, tsplit, cats) if ccfg.get("use_templates", True) else []
        result[split] = dict(pairs=nat + tp, template_fills=fills)
        cnt = Counter((p["source"], p["category"]) for p in nat + tp)
        log.info(f"[cf] {split}: {len(nat)} natural + {len(tp)} template pairs; by (source,category)={dict(cnt)}")
    return result


def proxy_coverage(lex: ProxyLexicon, df: pd.DataFrame) -> pd.DataFrame:
    """How often each (category, group) occurs and its offensive rate - a lexicon audit table."""
    rows = []
    for text, y in zip(df["text"], df["label"]):
        for c, g in tag_text(lex, text).items():
            rows.append(dict(category=c, group=g, label=y))
    if not rows:
        return pd.DataFrame(columns=["category", "group", "count", "positive_rate"])
    t = pd.DataFrame(rows)
    return (t.groupby(["category", "group"])["label"].agg(count="size", positive_rate="mean")
            .reset_index().sort_values(["category", "count"], ascending=[True, False]))
