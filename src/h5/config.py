"""YAML config loading with `extends:` inheritance and `--set a.b=c` overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides: Optional[List[str]] = None) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_update(load_config(path.parent / parent), cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
