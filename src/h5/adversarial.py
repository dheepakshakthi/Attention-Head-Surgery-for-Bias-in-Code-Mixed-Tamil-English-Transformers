"""Full adversarial debiasing baseline (gradient reversal; Ganin & Lempitsky 2015,
Zhang et al. 2018, Elazar & Goldberg 2018).

The whole encoder is fine-tuned from the pretrained checkpoint on the task loss
while one adversary per proxy category tries to predict the proxy group from the
[CLS] representation; a gradient-reversal layer makes the encoder remove that
information. Adversary batches are drawn category-balanced from comments tagged
with proxies plus their counterfactual variants (so every group is represented).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from .engine import CostTracker, encode, train_classifier
from .modeling import Adversary, HeadGatedClassifier, grad_reverse
from .utils import get_logger

log = get_logger()


def build_attribute_pools(records: List[dict], min_group_count: int = 10) -> Dict[str, dict]:
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["category"]][r["group"]].append(r["text"])
    pools = {}
    for c, groups in by.items():
        groups = {g: ts for g, ts in groups.items() if len(ts) >= min_group_count}
        if len(groups) >= 2:
            names = sorted(groups)
            pools[c] = {"groups": names, "texts": {g: groups[g] for g in names}}
    return pools


def train_adversarial(cfg: dict, train_records, dev_records, attr_records, num_labels: int, device,
                      tracker: CostTracker, eval_fn):
    acfg = cfg["adversarial"]
    model = HeadGatedClassifier.from_pretrained_encoder(cfg["model"]["name"], num_labels,
                                                       cfg["model"].get("dropout", 0.1)).to(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    pools = build_attribute_pools(attr_records, acfg.get("min_group_count", 10))
    if not pools:
        raise RuntimeError("No proxy category has >=2 groups with enough examples for the adversary")
    log.info("[adv] adversary categories: " + ", ".join(f"{c}({len(p['groups'])} groups)" for c, p in pools.items()))
    adv = Adversary(model.encoder.config.hidden_size, {c: len(p["groups"]) for c, p in pools.items()},
                    acfg.get("adversary_width", 256)).to(device)
    rng = random.Random(cfg.get("seed", 42))
    ce = nn.CrossEntropyLoss()
    tcfg = dict(cfg["train"], **acfg.get("train_overrides", {}))
    steps_per_epoch = math.ceil(len(train_records) / tcfg["batch_size"])
    total = steps_per_epoch * int(tcfg["epochs"])
    per_cat = max(2, acfg.get("adv_batch_size", 32) // len(pools))
    lam_max = float(acfg.get("lambda", 1.0))

    def extra_loss(m, batch, hidden, step):
        # DANN schedule: lambda ramps 0 -> lam_max
        p = min(1.0, step / max(1, total))
        lam = lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
        texts, cats, ys = [], [], []
        for c, pool in pools.items():
            for _ in range(per_cat):
                gi = rng.randrange(len(pool["groups"]))  # group-balanced sampling
                texts.append(rng.choice(pool["texts"][pool["groups"][gi]]))
                cats.append(c)
                ys.append(gi)
        b = encode(tok, texts, tcfg["max_length"], device)
        tracker.add_forward(b["attention_mask"], backward=True)
        h = m(**b, return_hidden=True)["hidden"]
        h = grad_reverse(h, lam)
        loss, logs, start = 0.0, {"lam": lam}, 0
        for c in pools:
            idx = [i for i, cc in enumerate(cats) if cc == c]
            hc = h[idx]
            yc = torch.tensor([ys[i] for i in idx], device=device)
            lc = ce(adv(hc.float(), c), yc)
            loss = loss + lc
        loss = loss / len(pools)
        logs["adv"] = float(loss.detach())
        return float(acfg.get("adv_weight", 1.0)) * loss, logs

    hist = train_classifier(model, tok, train_records, dev_records, tcfg, tracker, eval_fn,
                            extra_loss=extra_loss, extra_modules=[adv], seed=cfg.get("seed", 42), phase="adv_train")
    probe = probe_leakage(model, tok, pools, tcfg, device, cfg.get("seed", 42))
    return model, tok, {**hist, "adversary_categories": {c: p["groups"] for c, p in pools.items()},
                        "post_hoc_probe": probe}


def probe_leakage(model, tok, pools, tcfg, device, seed: int, n_per_group: int = 100, epochs: int = 50) -> dict:
    """Fresh linear probe on frozen [CLS]: how much proxy information is still decodable
    (not metered - it is an evaluation, not part of the method)."""
    rng = random.Random(seed)
    model.eval()
    out = {}
    for c, pool in pools.items():
        X, Y = [], []
        for gi, g in enumerate(pool["groups"]):
            ts = pool["texts"][g][:]
            rng.shuffle(ts)
            for t in ts[:n_per_group]:
                X.append(t)
                Y.append(gi)
        with torch.no_grad():
            hs = []
            for s in range(0, len(X), 64):
                b = encode(tok, X[s:s + 64], tcfg["max_length"], device)
                hs.append(model(**b, return_hidden=True)["hidden"].float())
            Hm = torch.cat(hs)
        Yt = torch.tensor(Y, device=device)
        perm = torch.randperm(len(Yt), generator=torch.Generator().manual_seed(seed)).to(device)
        n_tr = int(0.7 * len(Yt))
        tr, te = perm[:n_tr], perm[n_tr:]
        clf = nn.Linear(Hm.shape[1], len(pool["groups"])).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            nn.functional.cross_entropy(clf(Hm[tr]), Yt[tr]).backward()
            opt.step()
        acc = (clf(Hm[te]).argmax(-1) == Yt[te]).float().mean().item() if len(te) else float("nan")
        out[c] = {"probe_accuracy": acc, "chance": 1.0 / len(pool["groups"])}
    return out
