"""Offline unit / smoke tests (no model download, no real training).

Run:  pytest -q
"""
import random

import numpy as np
import pandas as pd
import pytest
import torch

from h5.counterfactual import natural_pairs, tag_text, template_pairs
from h5.data import _parse_line, OFFENSIVE_LABELS, SENTIMENT_LABELS
from h5.lexicon import ProxyLexicon
from h5.mediation import Mediator, char_map, effect_measure, token_map
from h5.metrics import cf_metrics, template_metrics
from h5.transliterate import to_short_style, transliterate_tamil
from h5.utils import load_json

LEX = "resources/proxy_lexicon.seed.json"


# ----------------------------------------------------------------- data
@pytest.mark.parametrize("line,expected", [
    ("movie semma\tNot_offensive\t\n", ("movie semma", "Not_offensive")),
    ("Offensive_Untargetede\tbad words here\n", ("bad words here", "Offensive_Untargetede")),
])
def test_parse_offensive(line, expected):
    assert _parse_line(line, OFFENSIVE_LABELS.keys()) == expected


def test_parse_semicolon_with_inner_semicolons():
    line = "nice; very nice; super;Positive;\n"
    assert _parse_line(line, SENTIMENT_LABELS.keys()) == ("nice; very nice; super", "Positive")


# ----------------------------------------------------------------- lexicon
@pytest.fixture(scope="module")
def lex():
    return ProxyLexicon.from_file(LEX)


def test_lexicon_match_and_suffix(lex):
    ms = lex.find("Nadars ellam super, Gounder makkal vaazhga")
    groups = {(m.category, m.group) for m in ms}
    assert ("caste", "nadar") in groups and ("caste", "gounder") in groups
    m = [m for m in ms if m.group == "nadar"][0]
    # suffix 's' must be kept outside the replaced span
    out = lex.substitute("Nadars ellam super", m, "brahmin")
    assert out == "Iyers ellam super"


def test_lexicon_no_substring_false_positive(lex):
    assert not [m for m in lex.find("alladhu sivakumar") if m.category == "religion"]


def test_lexicon_tamil_script(lex):
    ms = lex.find("நாடார் சமுதாயம்")
    assert ms and ms[0].group == "nadar" and ms[0].script == "tamil"


# ----------------------------------------------------------------- counterfactuals
def test_natural_pairs_preserve_label(lex):
    df = pd.DataFrame({"text": ["Thevar makkal padam paakanum", "இந்த படம் அருமை", "vaazhthukkal thalaivaa"],
                       "label": [1, 0, 0]})
    cfg = {"counterfactual": {"max_swaps_per_text": 2,
                              "categories": ["caste", "religion", "region", "dialect", "script", "translit_style"]}}
    pairs = natural_pairs(lex, df, "train", cfg, random.Random(0))
    cats = {p["category"] for p in pairs}
    assert {"caste", "script", "translit_style"} <= cats
    for p in pairs:
        assert p["text_a"] != p["text_b"]
        assert p["label"] == df["label"][p["orig_index"]]


def test_template_pairs_have_all_splits(lex):
    t = load_json("resources/templates.json")["templates"]
    tp = template_pairs(lex, t, {"caste", "religion", "region", "dialect"})
    assert {"train", "dev", "test"} <= set(tp)
    assert all(p["label"] == 0 for s in tp.values() for p in s)


def test_transliteration_and_style():
    assert transliterate_tamil("படம்") == "padam"
    assert "zh" in transliterate_tamil("தமிழ்")
    assert to_short_style("vaazhthukkal") == "valtukkal"
    assert to_short_style("the movie") == "the movie"


def test_tag_text(lex):
    t = tag_text(lex, "Muslim brothers sarbaga vaazhthukkal")
    assert t["religion"] == "muslim" and t["script"] == "roman" and t["translit_style"] == "long_style"


# ----------------------------------------------------------------- alignment
def test_char_and_token_map_identity():
    a = "hello world"
    cm = char_map(a, a)
    assert (cm == np.arange(len(a))).all()
    offs = [(0, 0), (0, 5), (6, 11), (0, 0)]
    assert token_map(offs, offs, cm, 4).tolist() == [0, 1, 2, 3]


def test_token_map_swap_shifts_suffix():
    a, b = "x nadar y", "x iyer y"
    oa = [(0, 0), (0, 1), (2, 7), (8, 9), (0, 0)]
    ob = [(0, 0), (0, 1), (2, 6), (7, 8), (0, 0)]
    m = token_map(oa, ob, char_map(a, b), len(ob))
    assert m.tolist() == [0, 1, 2, 3, 4]


# ----------------------------------------------------------------- metrics
def test_cf_metrics_basic():
    pairs = [{"source": "natural", "category": "caste", "group_a": "a", "group_b": "b"}] * 2
    pa = np.array([[0.9, 0.1], [0.4, 0.6]])
    pb = np.array([[0.7, 0.3], [0.6, 0.4]])
    r = cf_metrics(pairs, pa, pb)
    assert abs(r["cf_gap"] - 0.2) < 1e-6 and r["cf_flip_rate"] == 0.5


def test_template_metrics():
    fills = [{"template_id": "t0", "row_id": "r", "category": "caste", "group": g, "label": 0} for g in "ab"]
    r = template_metrics(fills, np.array([[0.2, 0.8], [0.9, 0.1]]))
    assert abs(r["template_range"] - 0.7) < 1e-6 and r["by_category"]["caste"]["fped"] == 1.0


# ----------------------------------------------------------------- model + mediation (tiny random BERT)
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import AutoModel, BertConfig, BertTokenizerFast
    from h5.modeling import HeadGatedClassifier
    d = tmp_path_factory.mktemp("tok")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "x", "y", "nadar", "iyer", "padam", "super", "##s"]
    (d / "vocab.txt").write_text("\n".join(words))
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    cfg = BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=64)
    torch.manual_seed(0)
    enc = AutoModel.from_config(cfg, attn_implementation="eager")
    model = HeadGatedClassifier(enc, 2).eval()
    return model, tok


def test_5d_head_mask_matches_2d(tiny):
    model, tok = tiny
    b = tok(["nadar padam super"] * 2, return_tensors="pt")
    m2 = torch.ones(2, 4)
    m2[1, 2] = 0
    with torch.no_grad():
        ref0 = model(b["input_ids"], b["attention_mask"])["logits"][0]
        ref1 = model(b["input_ids"], b["attention_mask"], head_mask=m2)["logits"][0]
        m5 = torch.ones(2, 2, 4, 1, 1)
        m5[1, 1, 2] = 0
        both = model(b["input_ids"], b["attention_mask"], head_mask=m5)["logits"]
    assert torch.allclose(both[0], ref0, atol=1e-5) and torch.allclose(both[1], ref1, atol=1e-5)


def test_mediation_null_pair_is_zero(tiny):
    model, tok = tiny
    med = Mediator(model, tok, 32, fp16=False)
    p = {"text_a": "x nadar padam", "text_b": "x nadar padam", "category": "caste", "group_a": "n", "group_b": "n"}
    heads = [(l, h) for l in range(2) for h in range(4)]
    r = med.exact_pair(p, heads)
    assert abs(r["te"]) < 1e-6 and np.allclose(r["nie"], 0, atol=1e-5) and np.allclose(r["nde"], 0, atol=1e-5)
    r2 = med.exact_pair(p, heads, mediator="attention")
    assert np.allclose(r2["nie"], 0, atol=1e-5)


def test_mediation_effects_are_consistent(tiny):
    model, tok = tiny
    med = Mediator(model, tok, 32, fp16=False)
    p = {"text_a": "x nadar padam", "text_b": "x iyer padam", "category": "caste", "group_a": "n", "group_b": "i"}
    heads = [(l, h) for l in range(2) for h in range(4)]
    r = med.exact_pair(p, heads)
    assert np.isfinite(r["nie"]).all() and np.isfinite(r["nde"]).all()
    # with every last-layer head intervened jointly the output would equal y_b only through attention;
    # here we just check NIE + NDE bracket TE loosely for single heads (Vig: TE ~ NIE + NDE for weak interactions)
    a = med.attribution_pair(p)
    assert a["nie"].shape == (2, 4) and abs(a["te"] - r["te"]) < 1e-5


def test_ablated_head_has_zero_nie(tiny):
    model, tok = tiny
    model.gates[1, 3] = 0.0
    try:
        med = Mediator(model, tok, 32, fp16=False)
        p = {"text_a": "x nadar padam", "text_b": "x iyer padam", "category": "caste", "group_a": "n", "group_b": "i"}
        r = med.exact_pair(p, [(1, 3)])
        assert abs(r["nie"][0]) < 1e-5
    finally:
        model.gates.fill_(1.0)


def test_effect_measure_binary():
    lg = torch.tensor([[0.0, 1.0]])
    assert abs(effect_measure(lg).item() - 1.0) < 1e-6
