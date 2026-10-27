"""Proxy-feature lexicon: loading, matching and counterfactual substitution.

A lexicon category (caste, religion, region, dialect) consists of *rows*. A row
aligns surface forms across groups that fill the same semantic slot in the
same script (e.g. deity names in Roman script). The first form of a group in a
row is its canonical form. A counterfactual is obtained by replacing a matched
form of group g with the canonical form of a different group g' **from the same
row**, so slot, script and agglutinated suffixes are preserved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .utils import load_json

TAMIL_CHAR = r"\u0B80-\u0BFF"


@dataclass
class Match:
    category: str
    group: str
    row_id: str
    slot: str
    script: str
    start: int
    end: int
    form: str


@dataclass
class Row:
    id: str
    slot: str
    script: str
    confidence: str
    forms: Dict[str, List[str]]

    def canonical(self, group: str) -> Optional[str]:
        f = self.forms.get(group)
        return f[0] if f else None


@dataclass
class Category:
    name: str
    groups: List[str]
    rows: List[Row] = field(default_factory=list)
    description: str = ""


class ProxyLexicon:
    def __init__(self, spec: dict, min_confidence: str = "low"):
        self.spec = spec
        order = {"low": 0, "medium": 1, "high": 2}
        floor = order[min_confidence]
        suffixes = spec.get("matching", {}).get("latin_suffixes", [])
        suf = "|".join(sorted((re.escape(s) for s in suffixes), key=len, reverse=True))
        self._suffix_re = f"(?:{suf})?" if suf else ""
        self.categories: Dict[str, Category] = {}
        self.rule_categories: Dict[str, dict] = spec.get("rule_based_categories", {})
        self._patterns = []  # (compiled, category, group, row)
        for cname, c in spec["categories"].items():
            cat = Category(cname, list(c["groups"]), description=c.get("description", ""))
            for r in c["rows"]:
                if order.get(r.get("confidence", "low"), 0) < floor:
                    continue
                row = Row(r["id"], r["slot"], r["script"], r.get("confidence", "low"),
                          {g: list(v) for g, v in r["forms"].items() if v})
                if len(row.forms) < 2:
                    continue
                cat.rows.append(row)
                for g, forms in row.forms.items():
                    for form in forms:
                        self._patterns.append((self._compile(form, row.script), cname, g, row))
            if cat.rows:
                self.categories[cname] = cat
        # longest forms first so 'iyengar' wins over 'iyer'-like overlaps
        self._patterns.sort(key=lambda p: -len(p[0].pattern))

    @classmethod
    def from_file(cls, path, min_confidence: str = "low") -> "ProxyLexicon":
        return cls(load_json(path), min_confidence=min_confidence)

    def _compile(self, form: str, script: str):
        esc = re.escape(form.lower()).replace(r"\ ", r"\s+")
        if script == "latin":
            return re.compile(rf"(?<![a-z]){esc}{self._suffix_re}(?![a-z])", re.IGNORECASE)
        return re.compile(rf"(?<![{TAMIL_CHAR}]){esc}")

    # ------------------------------------------------------------------
    def find(self, text: str, categories: Optional[List[str]] = None) -> List[Match]:
        taken = [False] * len(text)
        out: List[Match] = []
        for pat, cname, g, row in self._patterns:
            if categories and cname not in categories:
                continue
            for m in pat.finditer(text):
                s = m.start()
                form_len = self._form_len(m.group(0), row, g)
                e = s + form_len
                if any(taken[s:e]):
                    continue
                for k in range(s, e):
                    taken[k] = True
                out.append(Match(cname, g, row.id, row.slot, row.script, s, e, text[s:e]))
        out.sort(key=lambda m: m.start)
        return out

    @staticmethod
    def _form_len(matched: str, row: Row, group: str) -> int:
        """Length of the lexical form inside a match (suffix excluded)."""
        best = 0
        for f in row.forms[group]:
            m = re.match(re.escape(f.lower()).replace(r"\ ", r"\s+"), matched, re.IGNORECASE)
            if m:
                best = max(best, len(m.group(0)))
        return best or len(matched)

    def row(self, category: str, row_id: str) -> Row:
        for r in self.categories[category].rows:
            if r.id == row_id:
                return r
        raise KeyError(row_id)

    def substitute(self, text: str, match: Match, target_group: str) -> Optional[str]:
        row = self.row(match.category, match.row_id)
        repl = row.canonical(target_group)
        if repl is None or target_group == match.group:
            return None
        orig = text[match.start:match.end]
        if orig[:1].isupper():
            repl = repl[:1].upper() + repl[1:]
        return text[:match.start] + repl + text[match.end:]

    def group_index(self, category: str) -> Dict[str, int]:
        if category in self.categories:
            return {g: i for i, g in enumerate(self.categories[category].groups)}
        return {g: i for i, g in enumerate(self.rule_categories[category]["groups"])}

    def all_category_names(self) -> List[str]:
        return list(self.categories) + list(self.rule_categories)

    def rows_for_slot(self, slot: str, script: str) -> List[tuple]:
        out = []
        for cname, cat in self.categories.items():
            for r in cat.rows:
                if r.slot == slot and r.script == script:
                    out.append((cname, r))
        return out


def resolve_lexicon_path(cfg: dict) -> Path:
    """Prefer a native-speaker validated lexicon when it exists."""
    lcfg = cfg.get("lexicon", {})
    validated = Path(lcfg.get("validated_path", "resources/proxy_lexicon.validated.json"))
    if lcfg.get("prefer_validated", True) and validated.exists():
        return validated
    return Path(lcfg.get("seed_path", "resources/proxy_lexicon.seed.json"))
