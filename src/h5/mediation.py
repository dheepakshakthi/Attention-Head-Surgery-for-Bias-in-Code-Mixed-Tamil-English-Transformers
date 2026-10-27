"""Causal mediation analysis of attention heads (Vig et al., NeurIPS 2020).

Notation follows the paper. For a counterfactual pair (u = x_a, u' = x_b) that
differs only in a proxy feature (e.g. Nadar -> Iyer, Tamil script -> Roman):

* effect measure  y(u) = log-odds of the harmful class, log P(off|u) / (1 - P(off|u))
  (additive analogue of Vig's ratio p(anti)/p(stereo)),
* total effect    TE      = y(u') - y(u)
* natural indirect effect of head h
                  NIE_h   = y(u ; z_h <- z_h(u')) - y(u)
  (input u, only the mediator of head h is set to its value under u'),
* natural direct effect of head h
                  NDE_h   = y(u' ; z_h <- z_h(u)) - y(u)
  (input u', head h frozen at its value under u).

Two mediator types are supported:

* ``head_output`` - z_h is the head's context vectors (output of softmax(QK^T)V for head h,
  before W_O). This is exactly what head re-weighting scales, so NIE predicts what
  surgery removes.
* ``attention``   - z_h is the head's attention-weight matrix, Vig et al.'s
  "attention intervention" (values are still computed from the real input).

Because x_a and x_b tokenise to different lengths, mediator values are transferred
through a monotone **character alignment** (difflib) mapped onto sub-word offsets:
tokens outside the edited span map one-to-one, tokens inside it map proportionally.

Effects are **oriented** by sign(TE): an oriented NIE of +0.3 means "head h carries
0.3 log-odds of the pair's total counterfactual difference".

A cheap first-order screen (attribution patching, NIE_h ~ (z_h(u') - z_h(u)) . dy/dz_h)
ranks all heads with one forward/backward per pair; exact interventions are then run
on the top-M heads only (``screen_then_exact``).
"""
from __future__ import annotations

import difflib
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .engine import CostTracker
from .metrics import POS
from .utils import autocast_ctx, get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# effect measure
def effect_measure(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.shape[-1] == 2:
        return logits[:, POS] - logits[:, 1 - POS]
    lp = torch.log_softmax(logits, -1)[:, POS]
    return lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))


# ---------------------------------------------------------------------------
# alignment
def char_map(a: str, b: str) -> np.ndarray:
    """For every char position of a, the aligned char position in b (monotone)."""
    m = np.zeros(max(1, len(a)), dtype=np.int64)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            m[i1:i2] = np.arange(j1, j2)
        elif tag in ("replace", "delete"):
            n_a = i2 - i1
            n_b = max(1, j2 - j1)
            m[i1:i2] = j1 + np.minimum((np.arange(n_a) * n_b) // max(1, n_a), n_b - 1)
    return np.clip(m, 0, max(0, len(b) - 1))


def token_map(offsets_a: Sequence[Tuple[int, int]], offsets_b: Sequence[Tuple[int, int]],
              cmap: np.ndarray, len_b_tokens: int) -> np.ndarray:
    """Map each token index of a to a token index of b. Special tokens (offset (0,0))
    map to the special token at the same end of the sequence."""
    starts_b = np.array([s if e > s else -1 for s, e in offsets_b])
    content_b = np.where(starts_b >= 0)[0]
    out = np.zeros(len(offsets_a), dtype=np.int64)
    n_a = len(offsets_a)
    for i, (s, e) in enumerate(offsets_a):
        if e <= s:  # special token
            out[i] = 0 if i < n_a / 2 else len_b_tokens - 1
            continue
        if len(content_b) == 0:
            out[i] = 0
            continue
        cb = cmap[min(s, len(cmap) - 1)]
        # last b token whose start <= cb
        cand = content_b[starts_b[content_b] <= cb]
        out[i] = cand[-1] if len(cand) else content_b[0]
    return out


# ---------------------------------------------------------------------------
class HeadHooks:
    """Forward hooks on every layer's self-attention (head_output) or its attention-dropout
    (attention) for capturing and overwriting per-head mediator values."""

    def __init__(self, model, mediator: str = "head_output"):
        assert mediator in ("head_output", "attention")
        self.model = model
        self.mediator = mediator
        self.L, self.H, self.d = model.num_layers, model.num_heads, model.head_dim
        self.mode: Optional[str] = None
        self.captured: Dict[int, torch.Tensor] = {}
        self.retain_grad = False
        self.plan: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for l in range(self.L):
            sa = model.self_attention_module(l)
            mod = sa if mediator == "head_output" else sa.dropout
            self.handles.append(mod.register_forward_hook(self._make_hook(l)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    # z for head_output: [B, T, H, d] ; for attention: [B, H, T, T]
    def _make_hook(self, l: int):
        def hook(module, inputs, output):
            if self.mode is None:
                return None
            is_tuple = isinstance(output, tuple)
            t = output[0] if is_tuple else output
            if self.mode == "capture":
                if self.retain_grad and t.requires_grad:
                    t.retain_grad()
                self.captured[l] = t
                return None
            if self.mode == "intervene" and l in self.plan:
                bi, hi, src = self.plan[l]
                t = t.clone()
                if self.mediator == "head_output":
                    B, T, _ = t.shape
                    t4 = t.view(B, T, self.H, self.d)
                    t4[bi, :, hi, :] = src.to(t4.dtype)
                    t = t4.view(B, T, self.H * self.d)
                else:
                    t[bi, hi] = src.to(t.dtype)
                return (t,) + tuple(output[1:]) if is_tuple else t
            return None
        return hook

    def head_view(self, l: int) -> torch.Tensor:
        """Captured mediator of layer l in a uniform [B, H, ...] layout."""
        t = self.captured[l]
        if self.mediator == "head_output":
            B, T, _ = t.shape
            return t.view(B, T, self.H, self.d).permute(0, 2, 1, 3)  # [B,H,T,d]
        return t  # [B,H,T,T]


# ---------------------------------------------------------------------------
def _tokenize_one(tokenizer, text: str, max_length: int, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
    offs = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}, offs


def _align_mediator(z_src: torch.Tensor, idx: torch.Tensor, mediator: str) -> torch.Tensor:
    """z_src: [H, T_src, ...] captured on the source text; idx: [T_tgt] map tgt -> src positions.
    Returns the mediator value transported to the target's positions: [H, T_tgt, ...]."""
    if mediator == "head_output":
        return z_src[:, idx, :]
    a = z_src[:, idx][:, :, idx]  # [H, T_tgt, T_tgt]
    return a / a.sum(-1, keepdim=True).clamp_min(1e-9)


class Mediator:
    def __init__(self, model, tokenizer, max_length: int = 128, fp16: bool = True,
                 tracker: Optional[CostTracker] = None):
        self.model = model
        self.tok = tokenizer
        self.max_length = max_length
        self.fp16 = fp16
        self.tracker = tracker
        self.device = next(model.parameters()).device
        self.L, self.H = model.num_layers, model.num_heads

    def _fwd(self, batch, n_copies: int = 1):
        if self.tracker:
            self.tracker.add_forward(batch["attention_mask"].expand(n_copies, -1))
        with autocast_ctx(self.device, self.fp16):
            return self.model(**batch)["logits"]

    def _prepare(self, pair: dict):
        ba, oa = _tokenize_one(self.tok, pair["text_a"], self.max_length, self.device)
        bb, ob = _tokenize_one(self.tok, pair["text_b"], self.max_length, self.device)
        m_ab = token_map(oa, ob, char_map(pair["text_a"], pair["text_b"]), len(ob))
        m_ba = token_map(ob, oa, char_map(pair["text_b"], pair["text_a"]), len(oa))
        return ba, bb, torch.as_tensor(m_ab, device=self.device), torch.as_tensor(m_ba, device=self.device)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def exact_pair(self, pair: dict, heads: Sequence[Tuple[int, int]], mediator: str = "head_output",
                   chunk: int = 48, want_nde: bool = True) -> dict:
        """Exact NIE/NDE for the given (layer, head) list on one pair."""
        self.model.eval()
        ba, bb, m_ab, m_ba = self._prepare(pair)
        with HeadHooks(self.model, mediator) as hk:
            hk.mode = "capture"
            ya = effect_measure(self._fwd(ba))[0].item()
            za = {l: hk.head_view(l)[0].detach() for l in range(self.L)}
            yb = effect_measure(self._fwd(bb))[0].item()
            zb = {l: hk.head_view(l)[0].detach() for l in range(self.L)}
            te = yb - ya
            res = {"te": te, "y_a": ya, "y_b": yb, "nie": np.zeros(len(heads)), "nde": np.zeros(len(heads))}
            jobs = [("nie", ba, zb, m_ab)] + ([("nde", bb, za, m_ba)] if want_nde else [])
            for key, base, zsrc, idx in jobs:
                aligned = {l: _align_mediator(zsrc[l], idx, mediator) for l in {h[0] for h in heads}}
                hk.mode = "intervene"
                for s in range(0, len(heads), chunk):
                    hs = heads[s:s + chunk]
                    n = len(hs)
                    batch = {k: v.expand(n, -1) for k, v in base.items()}
                    plan = defaultdict(lambda: ([], [], []))
                    for k, (l, h) in enumerate(hs):
                        plan[l][0].append(k)
                        plan[l][1].append(h)
                        plan[l][2].append(aligned[l][h])
                    hk.plan = {l: (torch.tensor(b, device=self.device), torch.tensor(hh, device=self.device),
                                   torch.stack(src)) for l, (b, hh, src) in plan.items()}
                    y = effect_measure(self._fwd(batch, n_copies=n)).cpu().numpy()
                    res[key][s:s + n] = y - ya
                hk.plan, hk.mode = {}, None
        return res

    # --------------------------------------------------------------------
    def attribution_pair(self, pair: dict) -> dict:
        """First-order NIE estimate for all heads (head_output mediator), one fwd+bwd."""
        self.model.eval()
        ba, bb, m_ab, _ = self._prepare(pair)
        with HeadHooks(self.model, "head_output") as hk:
            hk.mode = "capture"
            with torch.no_grad():
                yb = effect_measure(self._fwd(bb))[0].item()
                zb = {l: hk.head_view(l)[0].detach().float() for l in range(self.L)}
            emb = self.model.encoder.get_input_embeddings()
            # grads w.r.t. activations are needed even if the encoder weights are frozen
            h = emb.register_forward_hook(lambda m, i, o: o if o.requires_grad else o.detach().requires_grad_(True))
            try:
                with torch.enable_grad():
                    if self.tracker:
                        self.tracker.add_forward(ba["attention_mask"], backward=True)
                    with autocast_ctx(self.device, self.fp16):
                        logits = self.model(**ba)["logits"]
                    y = effect_measure(logits)[0]
                    raws = [hk.captured[l] for l in range(self.L)]
                    grads = torch.autograd.grad(y, raws)
            finally:
                h.remove()
            ya = y.item()
            nie = np.zeros((self.L, self.H))
            for l in range(self.L):
                T = raws[l].shape[1]
                za = raws[l].detach().float().view(1, T, self.H, -1)[0].permute(1, 0, 2)
                g = grads[l].float().view(1, T, self.H, -1)[0].permute(1, 0, 2)
                delta = _align_mediator(zb[l], m_ab, "head_output") - za
                nie[l] = (delta * g).sum((1, 2)).cpu().numpy()
        return {"te": yb - ya, "nie": nie}


# ---------------------------------------------------------------------------
def select_pairs(pairs: List[dict], n_per_category: int, seed: int, categories=None,
                 min_abs_te: float = 0.0) -> List[dict]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for p in pairs:
        if categories and p["category"] not in categories:
            continue
        by[p["category"]].append(p)
    out = []
    for c in sorted(by):
        ps = by[c][:]
        rng.shuffle(ps)
        out.extend(ps[:n_per_category])
    return out


def aggregate(records: List[dict], L: int, H: int, heads: Optional[List[Tuple[int, int]]] = None) -> dict:
    """Aggregate per-pair effects into per-category [L,H] tables (oriented by sign(TE))."""
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
        by["__all__"].append(r)
    out = {}
    for c, rs in by.items():
        te = np.array([r["te"] for r in rs])
        sgn = np.sign(te)
        sgn[sgn == 0] = 1
        entry = {"n": len(rs), "mean_abs_te": float(np.abs(te).mean()), "mean_te": float(te.mean())}
        for key in ("nie", "nde"):
            if key not in rs[0]:
                continue
            vals = np.stack([np.asarray(r[key], dtype=np.float64) for r in rs])  # [n, k] or [n, L, H]
            if heads is not None:  # expand partial head lists to [L,H] with NaN
                full = np.full((len(rs), L, H), np.nan)
                for k, (l, h) in enumerate(heads):
                    full[:, l, h] = vals[:, k]
                vals = full
            ori = vals * sgn[:, None, None]
            entry[f"{key}_oriented"] = np.nanmean(ori, 0)
            entry[f"{key}_abs"] = np.nanmean(np.abs(vals), 0)
            # share of total effect carried by the head
            entry[f"{key}_share"] = np.nansum(ori, 0) / max(1e-9, np.abs(te).sum())
        out[c] = entry
    return out


def run_attribution(med: Mediator, pairs: List[dict], desc: str = "attribution") -> List[dict]:
    recs = []
    for p in tqdm(pairs, desc=desc, leave=False):
        r = med.attribution_pair(p)
        recs.append({"category": p["category"], "group_a": p["group_a"], "group_b": p["group_b"], **r})
    return recs


def run_exact(med: Mediator, pairs: List[dict], heads: List[Tuple[int, int]], mediator: str,
              chunk: int, want_nde: bool = True, desc: str = "mediation") -> List[dict]:
    recs = []
    for p in tqdm(pairs, desc=desc, leave=False):
        r = med.exact_pair(p, heads, mediator=mediator, chunk=chunk, want_nde=want_nde)
        recs.append({"category": p["category"], "group_a": p["group_a"], "group_b": p["group_b"],
                     "text_a": p["text_a"], "text_b": p["text_b"], **r})
    return recs


def category_weighted_score(agg: dict, key: str, categories: Sequence[str]) -> np.ndarray:
    """Combine per-category oriented effects; each category normalised by its mean |TE|
    so that one high-TE category cannot dominate the ranking."""
    tabs = []
    for c in categories:
        if c in agg and f"{key}_oriented" in agg[c]:
            tabs.append(np.nan_to_num(agg[c][f"{key}_oriented"]) / max(1e-6, agg[c]["mean_abs_te"]))
    if not tabs:
        raise RuntimeError("No mediation results for the requested categories")
    return np.mean(tabs, 0)


@torch.no_grad()
def proxy_attention_ratio(model, tokenizer, pairs: List[dict], heads: List[Tuple[int, int]],
                          max_length: int = 128, fp16: bool = True, tracker=None) -> Dict[str, float]:
    """For lexical pairs with a known proxy span: mean attention mass that each head puts on the
    proxy tokens, divided by the mass expected under uniform attention. >1 => the head reads the proxy."""
    device = next(model.parameters()).device
    sums = defaultdict(list)
    for p in pairs:
        if not p.get("span_a"):
            continue
        b, offs = _tokenize_one(tokenizer, p["text_a"], max_length, device)
        s, e = p["span_a"]
        span = [i for i, (o1, o2) in enumerate(offs) if o2 > o1 and o1 < e and o2 > s]
        if not span:
            continue
        if tracker:
            tracker.add_forward(b["attention_mask"])
        with autocast_ctx(device, fp16):
            att = model(**b, output_attentions=True)["attentions"]
        T = b["input_ids"].shape[1]
        for (l, h) in heads:
            a = att[l][0, h].float()  # [T,T]
            mass = a[:, span].sum(-1).mean().item()
            sums[f"L{l}.H{h}"].append(mass / (len(span) / T))
    return {k: float(np.mean(v)) for k, v in sums.items()}
