# H5 results: head surgery vs adversarial debiasing

## Verdict

- Fairness recovery (primary: CF gap): **-308.6%** (95% CI [-1133.7%, -165.8%]; threshold 80.0%) -> FAIL
- Cost ratio (FLOPs, surgery / adversarial): **4.7%** (target ~5.0%) -> PASS
- Wall-clock ratio: 1.5%; incl. full L x H analysis: 16.4%
- Macro-F1 drop: surgery -0.0063, adversarial 0.0031 -> PASS
- Names what was removed: yes
- **Claim supported: False**

## Bias metrics (test, lower is better)

| metric | baseline | surgery | adversarial | cda | recovery |
|---|---|---|---|---|---|
| CF gap (all pairs) | 0.0980 | 0.0816 | 0.1033 | - | -308.6% |
| CF gap (natural pairs) | 0.1012 | 0.0846 | 0.1077 | - | -254.1% |
| CF gap (template pairs) | 0.0347 | 0.0222 | 0.0162 | - | 67.4% |
| CF flip rate | 0.1157 | 0.1130 | 0.1170 | - | -200.0% |
| Template FPED | 0.1889 | 0.0556 | 0.0000 | - | 70.6% |
| Template range | 0.0852 | 0.0544 | 0.0447 | - | 76.0% |
| Group FPED (real) | 0.1565 | 0.1335 | 0.1477 | - | 261.8% |
| Group FNED (real) | 0.4034 | 0.3702 | 0.3934 | - | 331.4% |

## Task metrics

| split | method | accuracy | macro-F1 |
|---|---|---|---|
| hasoc_val | baseline | 0.8369 | 0.7879 |
| hasoc_val | surgery | 0.8466 | 0.7951 |
| hasoc_val | adversarial | 0.8380 | 0.7867 |
| test | baseline | 0.8162 | 0.7747 |
| test | surgery | 0.8277 | 0.7810 |
| test | adversarial | 0.8143 | 0.7716 |

## What the surgery removed

| head | gate | primary proxy category |
|---|---|---|
| L11.H3 | 0.055 | region |
| L4.H6 | 0.096 | dialect |
| L3.H11 | 0.562 | translit_style |
| L11.H0 | 0.064 | caste |
| L11.H7 | 0.063 | caste |
| L7.H9 | 0.186 | translit_style |
| L10.H1 | 0.069 | dialect |
| L11.H9 | 0.062 | translit_style |

Full per-head details (per-category NIE/NDE, proxy-attention ratio, top mediated swaps, examples): `outputs/surgery/surgery_report.json`.

## Proxy leakage probe (linear probe accuracy on [CLS]; lower = less proxy information)

- baseline: script=0.9833 (chance 0.5000), translit_style=0.5333 (chance 0.5000), dialect=0.4054 (chance 0.3333), caste=0.2652 (chance 0.1250), region=0.5208 (chance 0.2000), religion=0.1500 (chance 0.3333)
- surgery: script=0.9667 (chance 0.5000), translit_style=0.5000 (chance 0.5000), dialect=0.3649 (chance 0.3333), caste=0.2576 (chance 0.1250), region=0.4792 (chance 0.2000), religion=0.2500 (chance 0.3333)
- adversarial: script=1.0000 (chance 0.5000), translit_style=0.6000 (chance 0.5000), dialect=0.3108 (chance 0.3333), caste=0.2879 (chance 0.1250), region=0.1458 (chance 0.2000), religion=0.1500 (chance 0.3333)
