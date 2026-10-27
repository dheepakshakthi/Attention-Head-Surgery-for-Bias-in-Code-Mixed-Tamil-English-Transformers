"""Rule-based Tamil-script -> casual Roman ("Tanglish") transliteration and
Roman spelling-style perturbations.

These generate *script* and *transliteration-style* counterfactuals: content is
held fixed while only the surface encoding changes, which is exactly the
"script switching / transliteration style" proxy named in the problem.

The romanisation follows common YouTube-comment conventions (zh for ழ, th for
த, single 'n' for ந/ன/ண, 'd' for intervocalic ட), not ISO 15919.
"""
from __future__ import annotations

import re

VOWELS = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "ae", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
}
VOWEL_SIGNS = {
    "\u0BBE": "aa", "\u0BBF": "i", "\u0BC0": "ee", "\u0BC1": "u", "\u0BC2": "oo",
    "\u0BC6": "e", "\u0BC7": "ae", "\u0BC8": "ai", "\u0BCA": "o", "\u0BCB": "o", "\u0BCC": "au",
}
PULLI = "\u0BCD"
CONSONANTS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "d", "ண": "n", "த": "th", "ந": "n",
    "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
    "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஶ": "sh",
}
AYTHAM = "ஃ"
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")


def has_tamil_script(text: str) -> bool:
    return bool(TAMIL_RANGE.search(text))


def tamil_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if TAMIL_RANGE.match(c)) / len(letters)


def transliterate_tamil(text: str) -> str:
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == PULLI:
                out.append(base)
                i += 2
                continue
            if nxt in VOWEL_SIGNS:
                out.append(base + VOWEL_SIGNS[nxt])
                i += 2
                continue
            out.append(base + "a")
            i += 1
            continue
        if ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch == AYTHAM:
            out.append("h")
        elif ch in VOWEL_SIGNS or ch == PULLI:
            pass  # stray combining mark
        else:
            out.append(ch)
        i += 1
    s = "".join(out)
    # common casual simplifications: த்த -> "thth" is usually written "tt"/"th"
    s = s.replace("thth", "tth").replace("ngk", "ng")
    return s


# --- Roman spelling style -------------------------------------------------
_LONG_FEATURES = re.compile(r"(aa|ee|oo|zh|dh|th)", re.IGNORECASE)


def has_long_style(text: str) -> bool:
    """Phonetic 'long' spelling (doubled vowels, zh, dh/th digraphs) in a non-English Roman word."""
    if has_tamil_script(text):
        return False
    return any(_LONG_FEATURES.search(w) for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in _ENGLISH_KEEP)


def to_short_style(text: str) -> str:
    """Collapse to the 'short' casual style (vera, semma, padam, ...):
    aa->a, ee->i, oo->u, zh->l, dh->d, th->t (outside English 'the/that/this' etc.)."""
    def word_map(w: str) -> str:
        lw = w.lower()
        if lw in _ENGLISH_KEEP:
            return w
        r = re.sub("aa", "a", w)
        r = re.sub("AA", "A", r)
        r = re.sub("ee", "i", r)
        r = re.sub("oo", "u", r)
        r = re.sub("zh", "l", r)
        r = re.sub("dh", "d", r)
        r = re.sub(r"(?<!s)th", "t", r)
        return r

    return re.sub(r"[A-Za-z]+", lambda m: word_map(m.group(0)), text)


_ENGLISH_KEEP = {
    "the", "that", "this", "these", "those", "there", "their", "they", "then", "than",
    "with", "without", "good", "look", "looks", "need", "feel", "see", "seen", "too",
    "teeth", "mood", "food", "school", "cool", "boost", "keep", "free", "sweet", "week",
    "something", "nothing", "anything", "everything", "mother", "father", "brother",
    "thanks", "thank", "think", "thing", "things", "both", "month", "path", "bgm",
    "booking", "book", "moon", "soon", "speed", "agree", "deep", "meet", "street",
    "between", "together", "other", "another", "whether", "weather", "though", "through",
    "three", "third", "thousand", "throw", "thought", "maths", "truth", "youth",
    "brothers", "brother", "others", "mothers", "fathers", "teeth", "wheel", "door", "floor", "blood",
}
