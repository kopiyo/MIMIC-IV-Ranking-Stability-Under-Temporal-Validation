# MIMIC-IV — Ranking Stability Under Temporal Validation

Does the performance ranking of machine learning algorithms for ICU in-hospital
mortality prediction stay the same when models are evaluated by forward-in-time
temporal validation instead of random data splitting?

`mimic.ipynb` is the complete pipeline: data loading, preprocessing, sequence
tensor construction, hyperparameter tuning, final fitting, ranking comparison,
calibration analysis, figures and export.

---

## The design

Ten algorithms are trained twice, once under each evaluation design, and their
rankings compared.

| Family | Models |
|---|---|
| Linear | Logistic Regression (elastic net) |
| Bagged trees | Random Forest |
| Boosted trees | XGBoost, LightGBM, CatBoost |
| Neural (tabular) | MLP, FT-Transformer |
| Sequence | LSTM, GRU-D, Temporal Fusion Transformer |
| Reference (not ranked) | Modified SOFA, SAPS-II-lite |

| | Design A | Design B |
|---|---|---|
| Train | 70% random, all eras | E1–E3 (2008–2016) |
| Validation | 15% random | E4 (2017–2019) |
| Test | 15% random | E5 (2020–2022) |

Design A is how most benchmark papers evaluate. Design B is the deployment
condition: train on the past, predict the future. If the two designs produce
different rankings, the conventional benchmark is selecting the wrong model.

**Primary outcome:** Kendall's tau-b between the two rank orderings, with a
bootstrap confidence interval.

**Primary metric:** AUPRC, fixed in advance. At roughly 11% prevalence, AUROC is
inflated by the large true-negative count.

---

## Requirements

### Data access

MIMIC-IV is credentialed. You need a PhysioNet account, completed CITI
"Data or Specimens Only Research" training, and a signed data use agreement
before you can access it. **The derived parquet files cannot be redistributed** —
anyone reproducing this has to build them from their own credentialed access.

### Prerequisites

Two SQL extractions must have been run in BigQuery before the notebook will work:

| Script | Produces |
|---|---|
| `mimic_extraction.sql` | `mimic_icu_mortality_24h.parquet` — one row per ICU stay, 260 features |
| `mimic_sequences.sql` | `sequences.parquet` — long format, 22 channels × 24 hours |

A third file, `optuna_studies.db`, holds completed hyperparameter trials. It is
optional on a first run but saves many hours on any re-run.

### Environment

Written for **Kaggle** with a GPU accelerator. The notebook walks the whole
`/kaggle/input` tree rather than hard-coding paths, and picks the largest match
for each filename — which matters for `optuna_studies.db`, where the largest copy
is the one with the most completed trials.

A GPU is needed. The four neural models fall back to CPU and become impractically
slow: FT-Transformer alone took 134 minutes to tune on a T4.

Installed at runtime: `optuna`, `xgboost`, `lightgbm`, `catboost`. Already present
on the Kaggle image: `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`,
`matplotlib`. Python 3.12.

---

## Running it

1. Attach the datasets containing the three files above via **Add Input**.
2. Turn on the GPU: **Settings → Accelerator → GPU T4 x2**.
3. Run all cells from the top.

### Budget

`N_TRIALS = 30` Optuna trials per model per design, `N_SEEDS = 5` seeds for the
final fit, `SEED = 42`. The budget is identical for every algorithm, which is what
makes the ranking comparison fair.

### Resumability

Everything checkpoints. Optuna studies persist to SQLite; completed model/design
pairs are written to CSV as they finish and skipped on re-run. **If the session
disconnects, re-run from the top** — it picks up where it stopped. A disconnect
during tuning costs one trial, not the search.

The Optuna database is copied from the read-only input directory into
`/kaggle/working` first, because Optuna needs write access.

Download `results_backup.zip` from the Output tab before closing the session.
Kaggle working directories do not survive.

---

## Sections

| Section | What it does |
|---|---|
| 1 | Path discovery, budget constants, device check |
| 2 | Load parquet, leakage guards, build both split designs, preprocessing |
| 3 | Sequence tensors — `(n_stays, 24, n_channels)` plus mask and delta channels |
| 4 | Evaluation metrics: AUPRC, AUROC, Brier, calibration slope and intercept |
| 5 | Neural architectures: FT-Transformer, LSTM, GRU-D, TFT-lite |
| 6 | Model registry and the unified `fit_predict` dispatcher |
| 7 | Optuna tuning, all ten models, both designs |
| 8 | Final fit, 5 seeds each, predictions averaged |
| 9 | Clinical reference scores (modified SOFA, SAPS-II-lite) |
| 10 | **Primary analysis** — Kendall's tau and bootstrap CI |
| 11 | Figures |
| 12 | Calibration comparison |
| 13 | Sensitivity analyses |
| 14 | Export |

### Methodological commitments

These are deliberate and should not be "fixed" by a future editor:

- **Preprocessing statistics come from the training partition only.** Imputation
  medians, scaling statistics and category levels are fitted on train and applied
  unchanged to validation and test. In Design B that means 2008–2016 medians
  applied to 2020–2022 data. That is not a shortcut — it is the deployment
  condition under study. Fitting on the full dataset would leak the test era.
- **Class imbalance is handled by weighting, never resampling.** Resampling
  distorts predicted probabilities and would corrupt the calibration analysis.
- **Tuning uses the validation partition only.** In Design B that is era E4, so E5
  never influences hyperparameter choice.
- **Design A and B have different tabular widths** (332 vs 323 columns) because
  one-hot levels are fitted on each design's own training partition, and E1–E3
  contains fewer categorical levels than the full dataset.
- **Clinical scores are excluded from the ranking.** They are a fixed reference
  line, not competitors.

---

## Outputs

Written to `/kaggle/working/results`:

| File | Contents |
|---|---|
| `FINAL_SUMMARY.json` | Headline numbers — tau, CI, hypothesis verdict, cohort stats |
| `all_model_results.csv` | Every metric for all 20 model/design pairs |
| `all_results_with_scores.csv` | The above plus clinical reference scores |
| `ranking_comparison.csv` | Rank under each design, rank change, degradation |
| `table_calibration.csv` | Calibration slope, intercept and Brier by model |
| `all_predictions.npz` | Persisted test predictions — enables the bootstrap without refitting |
| `best_params.json` | Selected hyperparameters per model per design |
| `fig1_rank_comparison.png` | Slope chart of rank movement between designs |
| `fig2_degradation.png` | AUPRC change by algorithm and by family |
| `optuna_studies.db` | Trial history, reusable on re-run |

---

## Results from the last completed run

Cohort: 32,315 stays, 260 features (253 numeric, 7 categorical), 11.17%
prevalence. Sequence tensor 32,315 × 24 × 22 with 35.9% of cells observed.

Design A: 22,620 train / 4,847 val / 4,848 test.
Design B: 19,044 train / 7,321 val / 5,950 test.

| Model | Rank A | Rank B | AUPRC A | AUPRC B |
|---|---|---|---|---|
| XGBoost | 1 | 2 | 0.6256 | 0.6268 |
| LightGBM | 2 | 1 | 0.6159 | 0.6372 |
| CatBoost | 3 | 4 | 0.6121 | 0.6213 |
| FT-Transformer | 4 | 3 | 0.5920 | 0.6217 |
| MLP | 5 | 5 | 0.5909 | 0.6200 |
| RandomForest | 6 | 8 | 0.5576 | 0.5768 |
| LogisticRegression | 7 | 6 | 0.5551 | 0.6064 |
| GRU-D | 8 | 7 | 0.5459 | 0.5858 |
| TFT | 9 | 10 | 0.5414 | 0.5565 |
| LSTM | 10 | 9 | 0.5277 | 0.5660 |

Kendall tau-b = 0.778 (p = 0.0009). One model of ten moved two or more ranks
(Random Forest, 6 → 8). Best under Design A was XGBoost; best under Design B was
LightGBM. Both clinical reference scores sat far below every learned model
(AUPRC 0.29–0.35).

**Every model scored higher under Design B than Design A.** The notebook's own
caveat explains why: E5 prevalence (13.6%) exceeds training prevalence (10.6%),
and AUPRC rises mechanically with prevalence. Read the `degradation` column
alongside prevalence, not on its own.

---

## Known issues

These affect how the results should be interpreted. Work through them before
writing anything up.

### 1. The bootstrap draws independent resamples per model

In Section 10, the bootstrap loop draws a fresh index array inside the per-model
loop:

```python
for m in comp.index:
    y, p = preds[(m,'A')]; i = rng.integers(0, len(y), len(y))
    sa[m] = average_precision_score(y[i], p[i])
```

All Design A models share one test set in one row order, so a bootstrap replicate
should resample *patients once* and score every model on that same resample. Drawing
a separate `i` per model scores each model on a different set of patients, which
adds independent noise to each score and shuffles ranks at random. That biases tau
downward.

The symptom is visible in the output: the 95% CI is `[0.156, 0.778]`, with the
upper bound exactly equal to the point estimate. A correct bootstrap distribution
would normally straddle the estimate. Here the procedure appears able only to
degrade agreement, never improve it.

This matters because the headline conclusion — "CI excludes perfect agreement,
hypothesis supported" — rests on that upper bound. **Fix:** draw `i` once per
replicate per design, outside the model loop, then apply the same `i` to all ten
models. Re-run and see whether the conclusion survives.

### 2. "Degradation" is negative throughout

`degradation = auprc_A - auprc_B` is negative for every model, meaning Design B
scored higher. The figure axis label reads "AUPRC lost (A − B)" and the title
reads "Temporal degradation by algorithm", so both currently show gains labelled
as losses. Rename the variable and relabel the axes, or a reader will draw the
opposite conclusion from the figure.

### 3. The H3 calibration test ignores direction

Section 12 concludes "H3 supported" from `abs(d_slope) > abs(d_auprc)` — 5.9%
versus 4.4%. But mean calibration slope moved from 0.895 to 0.949, which is
*closer* to the ideal of 1.0. Calibration improved under Design B. The test
compares magnitudes without regard to sign, so it reports support for a hypothesis
that predicts degradation while the numbers show improvement. Rewrite the test to
compare distance from perfect calibration, not raw change.

### 4. The sensitivity analyses were not actually run

Section 13 executes SA1 only (ranking by AUROC, tau = 0.778, unchanged). SA2, SA3
and SA4 print partition sizes and instructions but do not run.

SA4 is the one that matters. E5 is 2020–2022. If ranking instability disappears
when E5 is removed, the honest conclusion is that a pandemic disrupts model
ranking — not that ordinary temporal drift does. That is a much narrower claim.
Run SA4 before making any general statement about temporal validation.

### 5. LightGBM Design A calibration slope is an outlier

0.322, against 0.75–1.27 for every other model, with an intercept near zero where
others sit around −1.3. Worth checking whether the tuned configuration produced
degenerate probability outputs before this goes in a table.

### 6. Stale header instructions

The opening cell says "Runtime → Change runtime type → T4" and "Everything
checkpoints to Drive", which are Colab instructions left over from an earlier
version. The code is Kaggle-specific throughout. Update the header so it matches.

### 7. The BigQuery fallback branch would fail

Section 3 falls back to pulling sequences from BigQuery if the cache is missing,
but `auth` is never imported and `BQ_PROJECT` is never defined in this notebook.
That branch has not been exercised in the Kaggle version. It works only because
the cached parquet is always present — either add the missing imports or delete
the branch.

---

## Caveats for the write-up

Carried over from the notebook, all of which belong in a limitations section:

- E5 prevalence (13.6%) exceeds training prevalence (10.6%). Higher prevalence
  mechanically raises AUPRC, so a model can look better under Design B for reasons
  unrelated to robustness.
- Design B trains on 19,044 stays against Design A's 22,620. SA3 is designed to
  isolate that, and has not been run.
- SOFA and SAPS-II are **modified**. Full SOFA needs vasopressor dose and
  PaO₂/FiO₂; full SAPS-II needs urine output and ventilation. All come from the
  `inputevents`, `outputevents` and `procedureevents` tables, excluded for roughly
  43% coverage in E5. They must be described as modified everywhere they appear.
- Organ-support features were excluded entirely, for the same coverage reason.
- The cohort is restricted to `year_offset = 0`, retaining 50.9% of first stays in
  E1 rising to 95.0% in E5. This is itself a time-varying selection effect and
  interacts with the temporal comparison.
- Only first ICU stays are included, so stay-level and patient-level partitions
  coincide and no group-aware splitting is needed.

---

## How to read the headline result

**tau well below 1 with a CI excluding 1** — random splitting can select a
different algorithm than temporal validation would. Report which models moved, in
which direction, and whether model capacity predicts movement.

**tau near 1 with a CI including 1** — conventional benchmarking selects the same
model despite its theoretical weakness. That is publishable, currently
unestablished, and useful.

At tau = 0.778 with one model of ten moving two ranks, and with the top two
positions swapping between two boosted-tree models whose AUPRC differs by 0.011,
the current evidence points toward substantial stability with a top-1 change. Fix
the bootstrap (issue 1) and run SA4 (issue 4) before committing to a stronger
claim than that.
