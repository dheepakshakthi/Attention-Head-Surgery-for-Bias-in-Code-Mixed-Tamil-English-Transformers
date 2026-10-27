"""Tokenisation, batched inference, generic fine-tuning loop and cost accounting.

Cost accounting
---------------
The "5 % of the training cost" claim must be checked in a hardware-independent
way, so every forward/backward pass is metered by :class:`CostTracker`:

* ``fwd_tokens`` / ``bwd_tokens``: non-padding tokens pushed through the encoder,
* ``approx_flops`` = 2·P·fwd_tokens + 4·P·bwd_tokens (Kaplan et al. 2020 rule of thumb,
  P = encoder parameters),
* ``wall_seconds``: CUDA-synchronised wall clock.

The comparison stage reports cost ratios on both FLOPs and wall clock.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .utils import autocast_ctx, get_logger, make_grad_scaler

log = get_logger()


# ---------------------------------------------------------------------------
class CostTracker:
    def __init__(self, n_params: int, name: str = ""):
        self.name = name
        self.n_params = int(n_params)
        self.fwd_tokens = 0
        self.bwd_tokens = 0
        self.wall_seconds = 0.0
        self.optimizer_steps = 0
        self.phases: Dict[str, dict] = {}
        self._t0 = None

    def add_forward(self, attention_mask: torch.Tensor, backward: bool = False):
        n = int(attention_mask.sum().item())
        self.fwd_tokens += n
        if backward:
            self.bwd_tokens += n

    @property
    def approx_flops(self) -> float:
        return 2.0 * self.n_params * self.fwd_tokens + 4.0 * self.n_params * self.bwd_tokens

    @contextmanager
    def phase(self, name: str):
        f0, b0 = self.fwd_tokens, self.bwd_tokens
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            dt = time.perf_counter() - t0
            self.wall_seconds += dt
            p = self.phases.setdefault(name, {"wall_seconds": 0.0, "fwd_tokens": 0, "bwd_tokens": 0})
            p["wall_seconds"] += dt
            p["fwd_tokens"] += self.fwd_tokens - f0
            p["bwd_tokens"] += self.bwd_tokens - b0

    def to_dict(self) -> dict:
        phases = {}
        for k, v in self.phases.items():
            phases[k] = dict(v, approx_flops=2.0 * self.n_params * v["fwd_tokens"] + 4.0 * self.n_params * v["bwd_tokens"])
        return dict(name=self.name, n_params=self.n_params, fwd_tokens=self.fwd_tokens,
                    bwd_tokens=self.bwd_tokens, approx_flops=self.approx_flops,
                    wall_seconds=self.wall_seconds, optimizer_steps=self.optimizer_steps,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    phases=phases)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


# ---------------------------------------------------------------------------
class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        texts = [b["text"] for b in batch]
        enc = self.tok(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "label" in batch[0]:
            out["labels"] = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return out


def encode(tokenizer, texts: Sequence[str], max_length: int, device) -> dict:
    enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


@torch.no_grad()
def predict_proba(model, tokenizer, texts: Sequence[str], max_length: int = 128, batch_size: int = 64,
                  head_mask: Optional[torch.Tensor] = None, tracker: Optional[CostTracker] = None,
                  fp16: bool = True, desc: Optional[str] = None) -> np.ndarray:
    """Softmax probabilities, shape [N, num_labels]. ``head_mask`` overrides the model gates."""
    device = next(model.parameters()).device
    model.eval()
    order = np.argsort([len(t) for t in texts])  # length bucketing => less padding
    out = np.zeros((len(texts), model.num_labels), dtype=np.float32)
    it = range(0, len(texts), batch_size)
    if desc:
        it = tqdm(it, desc=desc, leave=False)
    for s in it:
        idx = order[s:s + batch_size]
        b = encode(tokenizer, [texts[i] for i in idx], max_length, device)
        if tracker:
            tracker.add_forward(b["attention_mask"])
        with autocast_ctx(device, fp16):
            logits = model(**b, head_mask=head_mask)["logits"]
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
def make_loader(records: List[dict], tokenizer, max_length: int, batch_size: int, shuffle: bool, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=Collator(tokenizer, max_length),
                      generator=g if shuffle else None, num_workers=0)


def class_weights(labels: Sequence[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(np.float64)
    w = counts.sum() / (num_labels * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def train_classifier(model, tokenizer, train_records: List[dict], dev_records: List[dict], tcfg: dict,
                     tracker: CostTracker, eval_fn: Callable[[np.ndarray, np.ndarray], dict],
                     extra_loss: Optional[Callable] = None, extra_modules: Sequence[nn.Module] = (),
                     seed: int = 42, phase: str = "train") -> dict:
    """Full fine-tuning loop.

    ``extra_loss(model, batch, hidden, step) -> (loss, logs)`` lets adversarial
    debiasing plug in its adversary; it must meter its own passes on ``tracker``.
    Returns history and restores the best (dev macro-F1) weights in-place.
    """
    device = next(model.parameters()).device
    fp16 = tcfg.get("fp16", True)
    loader = make_loader(train_records, tokenizer, tcfg["max_length"], tcfg["batch_size"], True, seed)
    # Memory savers for small GPUs. XLM-R's 250k x 768 word-embedding matrix is ~70 % of its
    # parameters; freezing it removes its gradient and both AdamW moment buffers (~2.3 GB fp32).
    if tcfg.get("freeze_word_embeddings", False):
        model.encoder.get_input_embeddings().weight.requires_grad_(False)
        log.info("[train] word embeddings frozen")
    if tcfg.get("gradient_checkpointing", False):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info("[train] gradient checkpointing enabled")
    params = [p for p in model.parameters() if p.requires_grad]
    for m in extra_modules:
        params += [p for p in m.parameters() if p.requires_grad]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    named = list(model.named_parameters()) + [(f"extra{i}.{n}", p) for i, m in enumerate(extra_modules)
                                              for n, p in m.named_parameters()]
    groups = [
        {"params": [p for n, p in named if p.requires_grad and not any(k in n for k in no_decay)],
         "weight_decay": tcfg.get("weight_decay", 0.01)},
        {"params": [p for n, p in named if p.requires_grad and any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt_kwargs = {}
    if tcfg.get("optimizer_foreach") is not None:  # foreach=False avoids multi-tensor temp buffers
        opt_kwargs["foreach"] = bool(tcfg["optimizer_foreach"])
    opt = torch.optim.AdamW(groups, lr=float(tcfg["lr"]), **opt_kwargs)
    accum = int(tcfg.get("grad_accum", 1))
    total = math.ceil(len(loader) / accum) * int(tcfg["epochs"])
    max_steps = tcfg.get("max_steps")
    if max_steps:
        total = min(total, int(max_steps))
    sched = get_linear_schedule_with_warmup(opt, int(tcfg.get("warmup_ratio", 0.06) * total), total)
    scaler = make_grad_scaler(device, fp16)
    cw = class_weights([r["label"] for r in train_records], model.num_labels).to(device) \
        if tcfg.get("class_weighted", True) else None
    ce = nn.CrossEntropyLoss(weight=cw)

    best, best_state, history, step, bad_epochs = -1.0, None, [], 0, 0
    patience = tcfg.get("patience", 2)
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        for m in extra_modules:
            m.train()
        pbar = tqdm(loader, desc=f"{phase} ep{epoch}", leave=False)
        run_loss = 0.0
        with tracker.phase(phase):
            for i, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                tracker.add_forward(batch["attention_mask"], backward=True)
                with autocast_ctx(device, fp16):
                    out = model(batch["input_ids"], batch["attention_mask"], return_hidden=True)
                    loss = ce(out["logits"].float(), batch["labels"])
                    logs = {}
                    if extra_loss is not None:
                        el, logs = extra_loss(model, batch, out["hidden"], step)
                        loss = loss + el
                scaler.scale(loss / accum).backward()
                run_loss += loss.item()
                if (i + 1) % accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    step += 1
                    tracker.optimizer_steps += 1
                    pbar.set_postfix(loss=f"{run_loss / (i + 1):.4f}", **{k: f"{v:.3f}" for k, v in logs.items()})
                    if max_steps and step >= max_steps:
                        break
        # dev evaluation is metered separately (model selection overhead)
        with tracker.phase(f"{phase}_dev_eval"):
            probs = predict_proba(model, tokenizer, [r["text"] for r in dev_records], tcfg["max_length"],
                                  tcfg.get("eval_batch_size", 64), tracker=tracker, fp16=fp16)
        metrics = eval_fn(np.array([r["label"] for r in dev_records]), probs)
        metrics.update(epoch=epoch, step=step, train_loss=run_loss / max(1, len(loader)))
        history.append(metrics)
        log.info(f"[{phase}] epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                                          if isinstance(v, float)))
        if metrics["macro_f1"] > best:
            best, bad_epochs = metrics["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"[{phase}] early stopping")
                break
        if max_steps and step >= max_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_dev_macro_f1": best, "steps": step}
