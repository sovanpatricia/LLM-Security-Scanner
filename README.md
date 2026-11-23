# LLM Security Scanner (MIRPR 2026-2027)

Detects OWASP Top 10 for LLM Applications (2025) risks in Python and TypeScript code
that integrates large language models. The project has three stages so far:

1. **Synthetic snippets (Lab 3).** An intelligent algorithm trained on a small,
   generated dataset of short functions.
2. **Real application code.** A labelled set of vulnerable / repaired file pairs cut from
   four open-source LLM applications, used to check whether the scanner works on real code.
3. **Improvements (step 4).** Function-level analysis, real examples in training,
   hyperparameter tuning, regularization, ensembles and a small neural network, compared on
   performance, size/speed and explainability.

The full write-up is in `docs/lab3_report.pdf` (sections 1-5: synthetic data; section 6:
real application code; section 7: improvements).

## Layout

```
data/
  build_dataset.py        development set (199 snippets, 59 templates)
  build_val_newstyle.py   new-style validation set (63 snippets, 35 templates)
  build_testset.py        final synthetic test set (54 snippets, 32 templates)
  dataset.csv, val_newstyle.csv, testset.csv
  pilot/cases/            24 real-code cases: vulnerable.py + control.py + case.json each
  pilot/splits.json       split by application (validation: DVLA, streamlit-agent;
                          test: Vanna, PandasAI)
  pilot/README.md         annotation guide for case.json
  build_appset.py         builds appset.csv (48 files) from the cases
src/llmscan/
  rules.py                34 hand-written regex rules (baseline)
  features.py             comment stripping, code tokenizer, rule-hit features
  models.py               the learned models and their hyperparameter grids
  units.py                splits a file into functions, methods and assignments (step 4)
  context.py              context extraction for the real-code cases
  scan.py                 command-line scanner
experiments/
  run_experiments.py      Lab 3: tuning and choice on validation, one final test
  run_app_experiments.py  evaluation on real application code
  pilot/check_*.py        simulated attacks: each case is exploitable only in vulnerable.py
  pilot/eda_appset.py     exploratory analysis of the real-code dataset
  run_improvements.py     step 4, part 1: function-level input, real examples in training
  run_ensembles.py        step 4, part 2: soft voting, bagging, boosting, rules + model
  run_tuning.py           step 4, part 3: wider grids, L1/L2, MLP (Adam vs SGD)
  run_c_check.py          step 4, part 3b: SVM regularization below C = 0.03
  run_final_analysis.py   step 4, part 4: three-criteria comparison, choice, single test run
  save_final_model.py     saves the model chosen in step 4
  plot_step4.py           step-3 vs step-4 confusion matrices
reports/
  results.json, eda.json  Lab 3 results
  pilot/                  real-code and step-4 results (app_results.json,
                          improvements_part*.json, final_analysis.json, ...)
  figures/                all figures used in the report
models/
  scanner_model.joblib     Lab 3 model (TF-IDF + Logistic Regression); used by scan.py today
  scanner_model_v2.joblib  step-4 model (char n-grams + Linear SVM); not yet used by scan.py
examples/
  vulnerable_app.py       small deliberately vulnerable app for the demo
docs/
  lab3_report.tex, lab3_report.pdf
```

## Run

```bash
pip install -r requirements.txt

# Lab 3 (synthetic data)
python experiments/run_experiments.py

# Real application code
python data/build_appset.py
python experiments/pilot/check_variant_cases.py     # simulated attacks on the variant cases
python experiments/run_app_experiments.py

# Step 4 (improvements); every choice is made on the validation applications
python experiments/run_improvements.py
python experiments/run_ensembles.py
python experiments/run_tuning.py
python experiments/run_c_check.py
python experiments/run_final_analysis.py   # also scores the test applications
python experiments/save_final_model.py
python experiments/plot_step4.py

# Scanner
PYTHONPATH=src python -m llmscan.scan examples/vulnerable_app.py
PYTHONPATH=src python -m llmscan.scan path/to/your/repo --format json
```

All scripts run from the repository root, use fixed seeds and finish in seconds to a few
minutes on a laptop CPU. The scanner exits with code 1 when it reports findings, so it can
fail a CI job.

## Results

### Lab 3: synthetic snippets (macro-F1 over 9 classes)

Chosen on the new-style validation set, retrained on dev + validation, scored once on the
final test set.

| Model | Validation (used to choose) | Final test |
| --- | --- | --- |
| **TF-IDF tokens + Logistic Regression (chosen)** | **0.65** | **0.58** |
| TF-IDF char n-grams + Linear SVM | 0.64 | 0.64 |
| TF-IDF tokens + Random Forest | 0.40 | 0.39 |
| Hybrid: TF-IDF + rule hits + LogReg | 0.28 | 0.57 |
| Rules (baseline) | 0.18 | 0.28 |

### Real application code: before and after the improvements

Test applications Vanna and PandasAI (30 files, 15 vulnerable/repaired pairs). Macro-F1
over LLM02, LLM05, LLM06 and SAFE; *pair accuracy* = the vulnerable file gets its exact
label **and** its repaired version gets SAFE.

| Model | Macro-F1 | Pair accuracy | False alarms on safe files |
| --- | --- | --- | --- |
| Before (char SVM on whole files, synthetic training) | 0.27 | 0.07 (1/15) | 13/15 |
| **After step 4 (char SVM v2, chosen on validation)** | **0.46** | **0.40 (6/15)** | **6/15** |

The step-4 model reads single functions instead of whole files, is trained on the synthetic
data plus real functions from the validation applications (leave-one-app-out during
validation), and uses a stronger regularization (C = 0.03). It still misses data-leak cases
(LLM02), and the test applications had already been used once in the earlier evaluation;
see section 7 of the report for all limitations.

## Data and licenses

The synthetic snippets are generated from templates that reproduce patterns from the OWASP
LLM Top 10, framework documentation and public CVEs; they contain no real user data. The
real-code cases are excerpts of open-source applications (Apache-2.0 or MIT); each case
folder keeps the upstream URL, commit and license in its `case.json`.
