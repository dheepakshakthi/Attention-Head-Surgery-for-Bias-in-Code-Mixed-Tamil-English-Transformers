# H5 results: head surgery vs adversarial debiasing

## Verdict

- Fairness recovery (primary: CF gap): **53.1%** (95% CI [49.9%, 55.8%]; threshold 80.0%) -> FAIL
- Cost ratio (FLOPs, surgery / adversarial): **63.6%** (target ~5.0%) -> FAIL
- Wall-clock ratio: 124.0%; incl. full L x H analysis: 93.1%
- Macro-F1 drop: surgery 0.0000, adversarial 0.0000 -> PASS
- Names what was removed: yes
- **Claim supported: False**

## Bias metrics (test, lower is better)

| metric | baseline | surgery | adversarial | cda | recovery |
|---|---|---|---|---|---|
| CF gap (all pairs) | 0.0011 | 0.0010 | 0.0009 | - | 53.1% |
| CF gap (natural pairs) | 0.0019 | 0.0017 | 0.0016 | - | 54.1% |
| CF gap (template pairs) | 0.0001 | 0.0001 | 0.0001 | - | 24.6% |
| CF flip rate | 0.0000 | 0.0000 | 0.0000 | - | - |
| Template FPED | 0.0000 | 0.0000 | 0.0000 | - | - |
| Template range | 0.0002 | 0.0002 | 0.0002 | - | 36.4% |
| Group FPED (real) | 0.0000 | 0.0000 | 0.0000 | - | - |
| Group FNED (real) | 0.0000 | 0.0000 | 0.0000 | - | - |

## Task metrics

| split | method | accuracy | macro-F1 |
|---|---|---|---|
| hasoc_val | baseline | 0.7733 | 0.4361 |
| hasoc_val | surgery | 0.7733 | 0.4361 |
| hasoc_val | adversarial | 0.7733 | 0.4361 |
| test | baseline | 0.7533 | 0.4297 |
| test | surgery | 0.7533 | 0.4297 |
| test | adversarial | 0.7533 | 0.4297 |

## What the surgery removed

| head | gate | primary proxy category |
|---|---|---|
| L3.H1 | 0.613 | dialect |
| L2.H0 | 0.618 | translit_style |
| L0.H1 | 0.618 | religion |

Full per-head details (per-category NIE/NDE, proxy-attention ratio, top mediated swaps, examples): `outputs/surgery/surgery_report.json`.

## Proxy leakage probe (linear probe accuracy on [CLS]; lower = less proxy information)

- baseline: script=0.6875 (chance 0.5000), translit_style=0.4500 (chance 0.5000), region=0.5000 (chance 0.5000), caste=0.2121 (chance 0.1667)
- surgery: script=0.6875 (chance 0.5000), translit_style=0.4500 (chance 0.5000), region=0.3750 (chance 0.5000), caste=0.2121 (chance 0.1667)
- adversarial: script=0.6875 (chance 0.5000), translit_style=0.4500 (chance 0.5000), region=0.6250 (chance 0.5000), caste=0.2121 (chance 0.1667)
