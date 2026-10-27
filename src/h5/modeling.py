"""Sequence classifier over a BERT-family encoder with per-head gates.

Head gates are applied through Hugging Face's ``head_mask`` argument, which
multiplies the attention probabilities of every head (``[layers, heads]``).
Scaling a head's attention probabilities by g scales that head's contribution
to the layer output by g, so:

* g = 0  -> causal **ablation** of the head (used for localisation),
* 0<g<1  -> **re-weighting** (the debiasing method),
* g = 1  -> unchanged model.

No encoder weight is modified, so the surgery is a small, named, reversible edit.
Requires ``attn_implementation="eager"`` (SDPA kernels ignore head_mask).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .utils import ensure_dir, load_json, save_json


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam: float = 1.0):
    return GradReverse.apply(x, lam)


class HeadGatedClassifier(nn.Module):
    def __init__(self, encoder, num_labels: int, dropout: float = 0.1, model_name: str = ""):
        super().__init__()
        self.encoder = encoder
        self.model_name = model_name
        hid = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hid, num_labels)
        self.num_labels = num_labels
        self.num_layers = encoder.config.num_hidden_layers
        self.num_heads = encoder.config.num_attention_heads
        self.head_dim = hid // self.num_heads
        self.register_buffer("gates", torch.ones(self.num_layers, self.num_heads))

    # --- construction / persistence -----------------------------------
    @classmethod
    def from_pretrained_encoder(cls, name: str, num_labels: int, dropout: float = 0.1):
        try:
            enc = AutoModel.from_pretrained(name, attn_implementation="eager", add_pooling_layer=False)
        except TypeError:
            enc = AutoModel.from_pretrained(name, attn_implementation="eager")
        model = cls(enc, num_labels, dropout, model_name=name)
        model.check_architecture()
        return model

    def save(self, out_dir, tokenizer=None, extra_meta: Optional[dict] = None):
        out = ensure_dir(out_dir)
        self.encoder.save_pretrained(out / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(out / "encoder")
        torch.save({"classifier": self.classifier.state_dict(), "gates": self.gates.cpu()}, out / "head.pt")
        meta = {"num_labels": self.num_labels, "base_model": self.model_name}
        meta.update(extra_meta or {})
        save_json(meta, out / "meta.json")

    @classmethod
    def load(cls, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        meta = load_json(model_dir / "meta.json")
        enc = AutoModel.from_pretrained(model_dir / "encoder", attn_implementation="eager")
        model = cls(enc, meta["num_labels"], model_name=meta.get("base_model", ""))
        state = torch.load(model_dir / "head.pt", map_location="cpu")
        model.classifier.load_state_dict(state["classifier"])
        if "gates" in state:
            model.gates.copy_(state["gates"])
        model.check_architecture()
        return model.to(device), AutoTokenizer.from_pretrained(model_dir / "encoder")

    def check_architecture(self):
        try:
            _ = self.self_attention_module(0)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Head-level mediation needs a BERT-family encoder exposing "
                "encoder.layer[i].attention.self (BERT, mBERT, MuRIL, RoBERTa, XLM-R, ELECTRA)."
            ) from e

    def self_attention_module(self, layer: int) -> nn.Module:
        return self.encoder.encoder.layer[layer].attention.self

    # --- forward ------------------------------------------------------
    def forward(self, input_ids, attention_mask, head_mask: Optional[torch.Tensor] = None,
                token_type_ids=None, output_attentions: bool = False, return_hidden: bool = False):
        """``head_mask``: None (use ``self.gates``), ``[L, H]`` (multiplied with the gates) or
        a per-example ``[L, B, H, 1, 1]`` mask (multiplied with the gates)."""
        if head_mask is None:
            hm = self.gates
        elif head_mask.dim() == 2:
            hm = self.gates * head_mask
        elif head_mask.dim() == 5:
            hm = self.gates[:, None, :, None, None] * head_mask
        else:
            raise ValueError(f"head_mask must be 2-D or 5-D, got {tuple(head_mask.shape)}")
        if not hm.requires_grad and bool(torch.all(hm == 1)):
            hm = None  # fast path: identical to the unmodified encoder
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, head_mask=hm,
                      output_attentions=output_attentions, return_dict=True)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        h = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(h))
        res = {"logits": logits}
        if return_hidden:
            res["hidden"] = h
        if output_attentions:
            res["attentions"] = out.attentions
        return res


class Adversary(nn.Module):
    """One small MLP per protected-attribute category (gradient-reversal adversary)."""

    def __init__(self, hidden: int, category_sizes: dict, width: int = 256):
        super().__init__()
        self.heads = nn.ModuleDict({
            c: nn.Sequential(nn.Linear(hidden, width), nn.ReLU(), nn.Linear(width, n))
            for c, n in category_sizes.items()
        })

    def forward(self, h, category: str):
        return self.heads[category](h)


class GateParam(nn.Module):
    """Learnable gates g = max_gate * sigmoid(z) for selected heads, 1 elsewhere."""

    def __init__(self, num_layers: int, num_heads: int, trainable_mask: torch.Tensor,
                 init: float = 0.95, max_gate: float = 1.0):
        super().__init__()
        init = min(max(init / max_gate, 1e-3), 1 - 1e-3)
        z0 = torch.logit(torch.tensor(init))
        self.z = nn.Parameter(torch.full((num_layers, num_heads), float(z0)))
        self.register_buffer("mask", trainable_mask.float())
        self.max_gate = max_gate

    def forward(self):
        g = self.max_gate * torch.sigmoid(self.z)
        return self.mask * g + (1 - self.mask) * torch.ones_like(g)
