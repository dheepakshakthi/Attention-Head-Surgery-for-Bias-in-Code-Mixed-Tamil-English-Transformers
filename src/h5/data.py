"""Dataset loading for DravidianCodeMix (Tamil) and HASOC Tamil.

The files shipped in the workspace are not uniformly formatted:

* ``tamil_offensive_full_{train,dev,test}.csv``  ->  ``text<TAB>label<TAB>``
* ``tamil_offensive_full.csv``                   ->  ``label<TAB>text``
* ``tamil_sentiment_full_train.csv``             ->  ``text<TAB>label``
* ``tamil_sentiment_full_{dev,test}.csv``        ->  ``text;label;``
* HASOC ``tamil_offensive_speech_{train,val}.csv`` -> CSV with header ``label,comment``

``read_labeled_lines`` therefore detects the separator and the label position
per line, using the known label vocabulary, instead of trusting a fixed format.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils import get_logger

log = get_logger()

OFFENSIVE_LABELS = {
    "Not_offensive": 0,
    "Offensive_Untargetede": 1,
    "Offensive_Untargeted": 1,
    "Offensive_Targeted_Insult_Individual": 1,
    "Offensive_Targeted_Insult_Group": 1,
    "Offensive_Targeted_Insult_Other": 1,
    "not-Tamil": None,  # dropped
}
SENTIMENT_LABELS = {
    "Positive": 0,
    "Negative": 1,
    "Mixed_feelings": 2,
    "unknown_state": None,
    "not-Tamil": None,
}
SENTIMENT_LABEL_NAMES = ["Positive", "Negative", "Mixed_feelings"]
OFFENSIVE_LABEL_NAMES = ["Not_offensive", "Offensive"]

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200b", " ").replace("\u200c", "").replace("\u200d", "")
    return _WS.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Aggressive key used for cross-dataset leakage checks."""
    return re.sub(r"[^\w\u0B80-\u0BFF]+", "", normalize_text(text).lower())


def _parse_line(line: str, label_vocab: Iterable[str]) -> Optional[Tuple[str, str]]:
    vocab = set(label_vocab)
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    if "\t" in line:
        parts = [p for p in line.split("\t")]
        while parts and parts[-1].strip() == "":
            parts.pop()
        if len(parts) >= 2 and parts[-1].strip() in vocab:
            return "\t".join(parts[:-1]), parts[-1].strip()
        if len(parts) >= 2 and parts[0].strip() in vocab:
            return "\t".join(parts[1:]), parts[0].strip()
    stripped = line.rstrip().rstrip(";")
    if ";" in stripped:
        text, lab = stripped.rsplit(";", 1)
        if lab.strip() in vocab:
            return text, lab.strip()
    return None


def read_labeled_lines(path, label_vocab: Iterable[str]) -> pd.DataFrame:
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_line(line, label_vocab)
            if parsed is None:
                bad += 1
                continue
            rows.append({"text": parsed[0], "raw_label": parsed[1]})
    log.info(f"Read {len(rows)} rows from {Path(path).name} ({bad} unparsable lines skipped)")
    return pd.DataFrame(rows)


def _finalize(df: pd.DataFrame, mapping: Dict[str, Optional[int]], split: str, source: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["raw_label"].map(mapping)
    df = df[df["label"].notna() & (df["text"].str.len() > 0)].copy()
    df["label"] = df["label"].astype(int)
    df["split"] = split
    df["source"] = source
    return df.reset_index(drop=True)


def load_dravidian(data_dir, task: str = "offensive") -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    if task == "offensive":
        stem, mapping = "tamil_offensive_full", OFFENSIVE_LABELS
    elif task == "sentiment":
        stem, mapping = "tamil_sentiment_full", SENTIMENT_LABELS
    else:
        raise ValueError(f"Unknown task {task}")
    out = {}
    for split in ("train", "dev", "test"):
        path = data_dir / f"{stem}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[split] = _finalize(read_labeled_lines(path, mapping.keys()), mapping, split, "dravidiancodemix")
    return out


def load_hasoc(data_dir) -> Dict[str, pd.DataFrame]:
    """HASOC Tamil: binary 0 = not offensive, 1 = offensive."""
    data_dir = Path(data_dir)
    out = {}
    for split, name in (("train", "tamil_offensive_speech_train.csv"), ("val", "tamil_offensive_speech_val.csv")):
        path = data_dir / name
        if not path.exists():
            log.warning(f"HASOC file missing: {path}")
            continue
        df = pd.read_csv(path, dtype={"comment": str})
        df = df.dropna(subset=["comment", "label"]).rename(columns={"comment": "text"})
        df["raw_label"] = df["label"].astype(int).astype(str)
        df = _finalize(df[["text", "raw_label"]], {"0": 0, "1": 1}, split, "hasoc")
        out[split] = df
    return out


def subsample(df: pd.DataFrame, n: Optional[int], seed: int, stratify: bool = True) -> pd.DataFrame:
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    if stratify:
        frac = n / len(df)
        parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("label")]
        return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n, random_state=seed).reset_index(drop=True)


def prepare_datasets(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Return a dict of named evaluation/training frames according to the config.

    Keys: ``train``, ``dev``, ``test`` (primary dataset) and optionally
    ``hasoc_val`` (second evaluation split, leakage-filtered against train).
    """
    dcfg = cfg["data"]
    task = dcfg.get("task", "offensive")
    seed = cfg.get("seed", 42)
    primary = dcfg.get("primary", "dravidiancodemix")

    dcm = load_dravidian(dcfg["dravidian_dir"], task)
    hasoc = load_hasoc(dcfg["hasoc_dir"]) if (task == "offensive" and dcfg.get("hasoc_dir")) else {}

    if primary == "dravidiancodemix":
        frames = {"train": dcm["train"], "dev": dcm["dev"], "test": dcm["test"]}
        second = hasoc.get("val")
        second_name = "hasoc_val"
    elif primary == "hasoc":
        if not hasoc:
            raise RuntimeError("primary=hasoc but HASOC files were not found")
        tr = hasoc["train"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_dev = int(0.1 * len(tr))
        frames = {"train": tr.iloc[n_dev:].assign(split="train"),
                  "dev": tr.iloc[:n_dev].assign(split="dev"),
                  "test": hasoc["val"].assign(split="test")}
        second = dcm["test"]
        second_name = "dcm_test"
    else:
        raise ValueError(primary)

    # de-duplicate training data
    tr = frames["train"]
    tr = tr.assign(_k=tr["text"].map(dedup_key)).drop_duplicates("_k").drop(columns="_k")
    frames["train"] = tr.reset_index(drop=True)

    # Leakage filtering: HASOC Tamil and DravidianCodeMix share YouTube comments.
    train_keys = set(frames["train"]["text"].map(dedup_key))
    for name in ("dev", "test"):
        df = frames[name]
        overlap = df["text"].map(dedup_key).isin(train_keys).sum()
        log.info(f"{name}: {overlap}/{len(df)} texts also occur in train (kept, reported only)")
    if second is not None and dcfg.get("use_second_eval", True):
        keys = second["text"].map(dedup_key)
        mask = ~keys.isin(train_keys)
        log.info(f"{second_name}: removed {(~mask).sum()} of {len(second)} texts that leak from train")
        frames[second_name] = second[mask].reset_index(drop=True)

    frames["train"] = subsample(frames["train"], dcfg.get("max_train_examples"), seed)
    frames["dev"] = subsample(frames["dev"], dcfg.get("max_dev_examples"), seed)
    for k in list(frames):
        if k not in ("train", "dev"):
            frames[k] = subsample(frames[k], dcfg.get("max_eval_examples"), seed)
    for k, v in frames.items():
        log.info(f"[data] {k}: n={len(v)} label_dist={v['label'].value_counts().to_dict()}")
    return frames


def label_names(task: str) -> List[str]:
    return OFFENSIVE_LABEL_NAMES if task == "offensive" else SENTIMENT_LABEL_NAMES
