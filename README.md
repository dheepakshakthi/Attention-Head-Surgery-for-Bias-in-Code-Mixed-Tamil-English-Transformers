# H5: Attention-head surgery for bias in code-mixed Tamil-English transformers

This repo checks one claim. Take a multilingual classifier fine-tuned on Tamil-English (Tanglish) offensive-language data. Its bias towards **India-specific proxy features** (caste-marked names, religion markers, region, dialect, script and transliteration style) can be localised to a few attention heads with **causal mediation analysis** (Vig et al., NeurIPS 2020). Re-weighting just those heads ("surgery") should then:

1. recover **most** of the fairness gain of full **adversarial debiasing**,
2. at **about 5 %** of its training cost, and
3. **name** what was removed (which heads, which proxy categories, which group swaps).

The `compare` stage writes a PASS/FAIL verdict for each part, using held-out test data.

---

## 1. Pipeline at a glance

```
prepare ──► baseline ──► (analysis) ──► surgery ──┐
   │                                              ├──► evaluate ──► compare ──► REPORT.md, plots
   └──────► adversarial ──────────── (cda) ───────┘
```

| Stage | What it does | Main outputs (`outputs/…`) |
|---|---|---|
| `prepare` | Parses the inconsistent dataset files, cleans and de-duplicates them, and drops texts that leak across datasets. Tags proxies, then builds counterfactual (CF) pairs and template sets. | `data/*.jsonl`, `cf/{split}_pairs.jsonl`, `cf/{split}_fills.jsonl`, `cf/attr_*.jsonl`, `cf/lexicon_coverage_train.csv` |
| `baseline` | Fine-tunes XLM-R base (configurable) for binary offensive detection. Cost is metered. | `baseline/model`, `cost.json`, `train_history.json` |
| `analysis` | *Diagnostic only.* Runs the full Vig et al. TE / NIE / NDE for **all** L×H heads, per proxy category, with both mediator types. | `analysis/mediation_tables.npz`, `mediation_summary.json`, `cost.json` |
| `surgery` | Screens heads, runs exact mediation and ablation, selects a head set, then re-weights the head gates. Cost is metered. Saves only the gates (k numbers). | `surgery/model/gates.pt`, `surgery_report.json`, `selection.json`, `cost.json` |
| `adversarial` | Full fine-tune with a gradient-reversal adversary per proxy category. Cost is metered. | `adversarial/model`, `cost.json` |
| `cda` | *Optional extra baseline:* fine-tune on originals plus counterfactual augmentations. | `cda/…` |
| `evaluate` | Measures task quality and all fairness metrics on test, the second dataset, and a leakage probe. | `<method>/eval.json`, `<method>/cf_gaps_test.npy` |
| `compare` | Computes recovery, cost ratios, bootstrap CIs and the verdict, and draws plots. | `comparison.json`, `REPORT.md`, `plots/*.png` |

---

## 2. Setup

Requirements: Python 3.9–3.12 (the pinned `torch==2.4.1` has **no Python 3.13 wheel**; this machine's default `python3` is 3.13, so create the venv with e.g. `python3.11 -m venv .venv` or `uv venv -p 3.11`) and Linux. A CUDA GPU with at least 12 GB is recommended for the full run (XLM-R base, fp16). CPU is fine for the smoke config.

```bash
cd RTDL_Project_V2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # pinned versions
pip install -e .                    # makes `h5` importable and adds the `h5` command
```

Notes:
- The pinned `torch==2.4.1` wheel from PyPI targets CUDA 12.1. For another CUDA version, install torch first by following https://pytorch.org/get-started/previous-versions/ and then run `pip install -r requirements.txt`.
- On first use, Hugging Face downloads the model (`xlm-roberta-base`, about 1.1 GB). Set `HF_HOME` to choose the cache directory. For offline machines, pre-download the model and set `model.name` to the local path.
- If you skip `pip install -e .`, prefix commands with `PYTHONPATH=src`.

## 3. Running

### 3.1 Quick sanity run (minutes, CPU OK; the numbers mean nothing)

```bash
pytest -q                                   # offline unit tests (tiny random BERT, no downloads)
bash scripts/run_all.sh configs/smoke.yaml  # whole pipeline on ~600 examples, tiny model
```

### 3.2 Full experiment

```bash
bash scripts/run_all.sh                      # = configs/default.yaml (GPU >= 12 GB)
bash scripts/run_all.sh configs/gpu4gb.yaml  # ~4 GB GPU: frozen word embeddings, grad checkpointing, batch 8x4
```

or run one stage at a time, which makes it easy to resume:

```bash
h5 prepare     --config configs/default.yaml
h5 baseline    --config configs/default.yaml
h5 analysis    --config configs/default.yaml   # optional diagnostic (full L×H Vig analysis)
h5 surgery     --config configs/default.yaml
h5 adversarial --config configs/default.yaml
h5 evaluate    --config configs/default.yaml
h5 compare     --config configs/default.yaml
cat outputs/REPORT.md
```

`python -m h5.cli <stage> …` works the same as `h5 <stage> …`.

### 3.3 Overrides and variants

Any config key can be overridden with `--set key.sub=value`:

```bash
# different encoder, separate output dir
h5 baseline --set model.name=bert-base-multilingual-cased output_dir=outputs_mbert
# HASOC as the primary dataset (DravidianCodeMix test becomes the second evaluation)
bash scripts/run_all.sh configs/default.yaml --set data.primary=hasoc output_dir=outputs_hasoc
# ablation-only surgery (no gate training), or fixed down-weighting
h5 surgery --set surgery.reweight=zero
h5 surgery --set surgery.reweight=scale surgery.scale_alpha=0.3
# use Vig's attention-weight mediator for localisation
h5 surgery --set surgery.mediator=attention
# enable the CDA baseline
h5 cda --set cda.enabled=true && h5 evaluate --methods cda
# quick subset run on GPU
bash scripts/run_all.sh configs/default.yaml --set data.max_train_examples=5000 train.epochs=2
```

Approximate runtimes on one T4 or A10-class GPU with the default config: baseline and adversarial take about 30–60 min each, surgery a few minutes, analysis 10–30 min (both mediators), and evaluate about 5–10 min. **None of these were measured; the code has not been executed.**

---

## 4. Data

Only the Tamil files in the workspace are used.

| File | Format (detected per line) | Use |
|---|---|---|
| `DravidianCodeMix/tamil_offensive_full_{train,dev,test}.csv` | `text<TAB>label<TAB>` | **Primary task.** `Not_offensive` maps to 0. All five `Offensive_*` labels map to 1. `not-Tamil` rows are dropped. |
| `HASOC_Tamil_offensive/tamil_offensive_speech_{train,val}.csv` | CSV with header `label,comment` (0/1) | **Second evaluation** (`hasoc_val`), or the primary dataset with `data.primary=hasoc`. |
| `DravidianCodeMix/tamil_sentiment_full_*` | TSV or `text;label;` | Optional `data.task=sentiment`; the harmful class is `Negative`. |

The parser decides separator and label position per line against the known label vocabulary. This handles reversed columns and semicolons inside the text. Training data is de-duplicated. The two datasets share YouTube comments, so any second-dataset text whose normalised form occurs in train is **removed**, and the number removed is logged.

---

## 5. Proxy-feature lexicon and counterfactual protocol

### 5.1 Lexicon (`resources/proxy_lexicon.seed.json`)

| Category | Groups | Mechanism |
|---|---|---|
| `caste` | thevar, gounder, nadar, vanniyar, brahmin (Iyer/Iyengar), dalit (Pallar/Paraiyar/Devendrakula…), chettiar, mudaliar | lexical, Roman and Tamil script |
| `religion` | hindu, muslim, christian. Three slots: deity, identity noun, place of worship. | lexical |
| `region` | chennai, madurai, kongu (Kovai/Erode/Salem…), nellai, jaffna | lexical |
| `dialect` | standard, Madras Bashai, Kongu, Jaffna. Aligned word variants such as *enna/inna*, *sollunga/sollungo*, *nanba/machan/maapla*. | lexical |
| `script` | Tamil script vs Roman | rule-based transliteration (`transliterate.py`) |
| `translit_style` | long phonetic (*vaazhthukkal*) vs short casual (*valtukkal*) | rule-based respelling |

Design choices:
- Each **row** aligns forms that fill the same slot in the same script. A swap therefore replaces a deity with a deity in the same script, and agglutinative suffixes are kept (*Nadars* becomes *Iyers*).
- Ambiguous forms are left out on purpose and documented in `notes`: `pillai` also means "child", `alla` means "not", and `siva` is a common personal name.
- The lexicon contains only identity terms. It has **no slurs**.

> **Status: PROVISIONAL SEED.** The engineering team compiled the seed from public knowledge and from reading the corpus. Dialect rows have *low* confidence. Before the lexicon is reported as a contribution, native speakers should validate it:
> ```bash
> h5 lexicon-export                       # -> outputs/lexicon/annotation_sheet.csv (with corpus counts)
> # 2-3 native annotators fill keep(y/n), corrected_form, is_offensive_or_slur(y/n), notes
> h5 lexicon-merge --sheets a1.csv a2.csv a3.csv   # majority vote, slur flags veto
> ```
> If `resources/proxy_lexicon.validated.json` exists, every stage uses it automatically. Re-run from `prepare` after merging.

### 5.2 Counterfactual augmentation protocol (`counterfactual.py`)

1. **Natural pairs.** For every real comment, the first matched proxy of each category is swapped to at most `max_swaps_per_text` other groups from the same lexicon row. One intervention per pair means the effect can be attributed to that proxy. The label is kept. Comments that are mostly Tamil script also get a **script** twin (transliterated), and long-style Roman comments get a **style** twin.
2. **Template pairs.** `resources/templates.json` holds neutral comments (gold label: not offensive) with an `{X}` slot, in the style of Vig et al.'s templates. Every group of the matching rows fills the slot. Templates are split train/dev/test **round-robin within each (slot, script)**, so test templates are never seen by localisation, selection or gate training.
3. **Caps.** Pairs are capped per category (`max_pairs_per_category_*`), so frequent categories such as script do not swamp rare ones.
4. **Attribute records** (text, category, group) come from tagged comments plus their counterfactual twins. They feed the adversary and the leakage probe.

`cf/lexicon_coverage_train.csv` reports, for each (category, group), the count and the **offensive base rate**. This is the dataset-level correlation that a model can pick up.

---

## 6. Method

### 6.1 Model (`modeling.py`)
The classifier is a BERT-family encoder with a linear head on [CLS]. Each head has a **gate** `g ∈ [0,1]` (an `L×H` buffer) that is passed as Hugging Face `head_mask`. The mask multiplies the head's attention probabilities, so it scales the head's contribution. `g=0` ablates the head, `0<g<1` re-weights it, and `g=1` leaves the model unchanged. Surgery never touches encoder weights; its result is k numbers. Eager attention is forced because SDPA kernels ignore `head_mask`.

### 6.2 Causal mediation, following Vig et al. 2020 (`mediation.py`)
For a counterfactual pair (u, u′), the effect measure is y(u) = log-odds of the offensive class. This is the additive analogue of Vig's ratio p(anti)/p(stereo).

| Paper | Here |
|---|---|
| Total effect | TE = y(u′) − y(u) |
| Natural indirect effect of mediator z_h | NIE_h = y(u; z_h ← z_h(u′)) − y(u) |
| Natural direct effect | NDE_h = y(u′; z_h ← z_h(u)) − y(u) |
| Neuron / attention interventions | Mediator `head_output`: the head's context vectors (exactly what surgery scales). Mediator `attention`: the head's attention weights (Vig's attention intervention, hooked on the attention dropout). |
| Templates × professions | Templates × proxy groups, plus natural corpus pairs |

u and u′ tokenise to different lengths, so mediator values are carried across with a **monotone character alignment** (difflib) projected onto sub-word offsets. Attention maps are re-normalised after alignment. Effects are **oriented** by sign(TE) and summarised as the mean oriented NIE and NDE, and as the *share of TE* each head carries. Categories are normalised by their mean |TE| before being combined.

### 6.3 Head surgery (`surgery.py`)
Data hygiene: localisation uses **train** pairs, selection uses **dev** pairs, and every reported number uses **test** pairs.

1. **Screen.** Attribution patching gives a first-order NIE estimate for all L×H heads with one forward/backward per pair.
2. **Exact.** Vig NIE and NDE are computed for the top-M heads (default 24) by batched interventions.
3. **Ablate.** Each candidate is zero-ablated on dev, recording Δ CF-gap and Δ macro-F1.
4. **Select.** Heads are ranked by (1−w)·z(NIE) + w·z(ablation benefit). Heads whose solo ablation hurts F1 or increases bias are dropped. Prefixes k = 1…k_max are evaluated, and the smallest set that reaches 90 % of the best bias reduction within the F1 budget is kept.
5. **Re-weight.** In `learned` mode (the default), only the k gates are optimised, for 150 steps, with CE(task) + λ·(y(a) − y(b))² on train CF pairs. `zero` and `scale` modes are also available.
6. **Name what was removed.** `surgery_report.json` lists, for each head: its gate, its primary proxy category, per-category NIE/NDE and share of TE, the dev ablation effect, the **proxy-attention ratio** (attention mass on proxy tokens relative to uniform; above 1 means the head reads the proxy), its top mediated group swaps (e.g. `caste:nadar->brahmin`), and its strongest example pairs.

### 6.4 Adversarial baseline (`adversarial.py`)
This is a full fine-tune from the pretrained encoder, the standard alternative pipeline. One MLP adversary per proxy category predicts the group from [CLS] through a gradient-reversal layer, with the DANN λ ramp. Adversary batches are balanced across categories and groups. The same training hyper-parameters as the baseline are used, so the adversarial run is a fair "full debiasing" reference.

### 6.5 Cost accounting (`engine.CostTracker`)
Every forward and backward pass of every method is metered, including dev evaluation inside training and all of the surgery's localisation and selection passes:
- `approx_flops = 2·P·fwd_tokens + 4·P·bwd_tokens` (non-padding tokens, P = encoder parameters). This is hardware-independent and is the **primary** cost.
- CUDA-synchronised wall clock, broken down by phase (`screen`, `exact`, `ablate`, `select`, `gates`, `report`).

The cost ratio is surgery / adversarial. `compare` also reports the ratio *including* the optional full `analysis` stage, because a diagnostic L×H map is often wanted as well.

### 6.6 Metrics (`metrics.py`)
- **Task:** accuracy, macro-F1 and offensive-class F1 on `test` and `hasoc_val`.
- **Primary bias score B:** CF gap, the mean |P(off|u) − P(off|u′)| over test CF pairs, natural plus template.
- **Secondary bias scores:** CF flip rate; template FPED (Σ_g |FPR_g − FPR|, Dixon et al. 2018) and template range (max−min P(off) across fills); FPED, FNED and demographic-parity gap across proxy groups in *real* test comments; and a leakage probe, i.e. linear-probe accuracy for each proxy category from [CLS], compared with chance.
- **Claim quantities:** gain G(m) = B(baseline) − B(m), **recovery = G(surgery)/G(adversarial)** with a 95 % paired-bootstrap CI over test pairs, and **cost ratio** as above.

### 6.7 Verdict (`comparison.json → verdict`, top of `REPORT.md`)
The claim counts as supported only if all of the following hold:
- `recovers_most`: recovery ≥ 0.8 (`comparison.recovery_threshold`),
- `adversarial_gain_positive`: adversarial training actually reduced B (otherwise recovery is meaningless),
- `cost_about_5pct`: FLOP ratio ≤ 5 % × tolerance 2 = 10 % (`cost_target`, `cost_tolerance`),
- `surgery_task_preserved`: macro-F1 drop ≤ 0.02,
- `names_what_was_removed`: the surgery report is non-empty.

The thresholds are explicit config values, so a stricter or looser reading of "most" or "about 5 %" is a one-line change, and the raw numbers are always reported.

---

## 7. Outputs

```
outputs/
├── REPORT.md                 # verdict + tables (start here)
├── comparison.json           # every number in the report
├── plots/                    # bias_metrics.png, cost.png, *_nie_heatmaps.png
├── data/  cf/                # prepared data, CF pairs, lexicon coverage
├── baseline/ adversarial/ cda/   model/, cost.json, train_history.json, eval.json
├── surgery/                  # model/gates.pt (+ pointer to baseline), surgery_report.json, selection.json
├── analysis/                 # mediation_tables.npz, mediation_summary.json (top-10 heads per category)
└── run.log, config_<stage>.json
```

## 8. Code layout

```
src/h5/
  data.py            robust loaders, cleaning, leakage filter
  lexicon.py         proxy lexicon: matching (suffix-aware, script-aware), substitution
  transliterate.py   Tamil→Roman transliteration, long→short Roman respelling
  counterfactual.py  CF protocol: natural pairs, templates, attribute records, coverage
  modeling.py        gated classifier, gradient reversal, adversary, gate parametrisation
  engine.py          tokenisation, inference, training loop, CostTracker
  mediation.py       Vig TE/NIE/NDE (head_output & attention mediators), attribution screen, alignment
  surgery.py         localise → ablate → select → re-weight → named report
  adversarial.py     adversarial debiasing baseline + leakage probe
  metrics.py         task + fairness metrics, bootstrap
  evaluate.py        held-out evaluation of one model
  compare.py         recovery / cost / verdict / report / plots
  cli.py             stage runner; config.py, utils.py helpers
configs/  default.yaml, smoke.yaml          resources/  lexicon + templates
scripts/run_all.sh                          tests/test_pipeline.py
```

## 9. Limitations and threats to validity

- **Lexicon validity.** The seed lexicon has not been validated, and the dialect rows in particular are low-confidence. Treat results as preliminary until the annotation round-trip (§5.1) is done.
- **Counterfactual naturalness.** Lexical swaps can produce unusual combinations, such as a deity name in a context typical of another religion. Rule-based transliteration only approximates real Tanglish spelling. Template pairs are fully controlled but synthetic. Reporting natural and template pairs separately (`natural_only`, `template_only`) shows whether the conclusions depend on either source.
- **Choice of B.** Recovery is defined on the CF gap. The secondary metrics are reported alongside it, and recovery is shown for each of them, because surgery that is optimised on CF consistency can look better on the CF gap than on real-data FPED/FNED.
- **Cost comparison.** Adversarial debiasing here is a full fine-tune. Surgery's cost excludes the baseline training it starts from, since that model already exists in the deployment scenario H5 assumes. `surgery_vs_baseline_flops` is also reported.
- **Seeds.** One seed by default. For a publishable claim, repeat with `--set seed=…` and `output_dir=…` for 3–5 seeds and report the spread.
- **Mediation assumptions.** NIE and NDE per head assume the other heads are held at natural values. Interactions between heads are covered only by the prefix selection step, not by joint mediation.
- **Status.** This code was written but **not executed** by its author. The unit tests and the smoke config are the intended first checks.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Head-level mediation needs a BERT-family encoder…` | Use XLM-R, mBERT, MuRIL or another encoder that exposes `encoder.layer[i].attention.self`. |
| CUDA OOM on a ~4 GB GPU (e.g. in `adam.py … exp_avg`) | Full XLM-R fine-tuning cannot fit in 4 GB (weights + grads + AdamW ≈ 4.5 GB). Use `bash scripts/run_all.sh configs/gpu4gb.yaml`, which freezes the word embeddings, enables gradient checkpointing and uses batch 8 × accumulation 4. |
| CUDA OOM (larger GPU) | Lower `train.batch_size` (set `train.grad_accum` to compensate), `train.max_length`, or `surgery.intervention_chunk` / `analysis.intervention_chunk`. |
| Warning `lr_scheduler.step() before optimizer.step()` on the first step | Harmless: the fp16 GradScaler skips the first step(s) while it calibrates the loss scale. |
| `RuntimeError: No proxy category has >=2 groups…` | The lexicon matched too little. Check `cf/lexicon_coverage_train.csv` and lower `adversarial.min_group_count`. |
| Recovery is `-` / `adversarial_gain_positive=false` | Adversarial training did not reduce B. Increase `adversarial.lambda` or the number of epochs. |
| NaN loss with fp16 | Use `--set train.fp16=false`. |

## Reference
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., Shieber, S. *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS 2020.
