# LLM Security Scanner (MIRPR 2026-2027)

Detects OWASP Top 10 for LLM Applications (2025) risks in Python and TypeScript code
that integrates large language models. Lab 3 stage: an intelligent algorithm trained on
a small dataset.

## Layout

```
data/
  build_dataset.py     generates the development set (199 snippets, 59 templates)
  build_val_newstyle.py  generates the new-style validation set (63 snippets, 35 templates)
  build_testset.py     generates the final test set (54 snippets, 32 templates)
  dataset.csv, val_newstyle.csv, testset.csv
src/llmscan/
  rules.py             34 hand-written regex rules (baseline)
  features.py          comment stripping, code tokenizer, rule-hit features
  models.py            the four learned models and their hyperparameter grids
  scan.py              command-line scanner that uses the trained model
experiments/
  run_experiments.py   EDA, tuning and model choice on validation, one final test, CV, figures
reports/
  eda.json, results.json, figures/
models/
  scanner_model.joblib the chosen model (TF-IDF + Logistic Regression), trained on dev + validation
examples/
  vulnerable_app.py    small deliberately vulnerable app for the demo
```

## Run

```bash
pip install -r requirements.txt
python data/build_dataset.py        # optional: data/*.csv are already committed
python data/build_val_newstyle.py
python data/build_testset.py
python experiments/run_experiments.py   # ~20 s on a laptop CPU
PYTHONPATH=src python -m llmscan.scan examples/vulnerable_app.py
PYTHONPATH=src python -m llmscan.scan path/to/your/repo --format json
```

The scanner exits with code 1 when it reports findings, so it can fail a CI job.

## Lab 3 results (macro-F1 over 9 classes)

Protocol: tune and choose on the new-style validation set, retrain on dev + validation,
score the chosen model once on the final test set (see `experiments/run_experiments.py`).

| Model | New-style validation (used to choose) | Final test |
| --- | --- | --- |
| **TF-IDF tokens + Logistic Regression (chosen)** | **0.65** | **0.58** |
| TF-IDF char n-grams + Linear SVM | 0.64 | 0.64 |
| TF-IDF tokens + Random Forest | 0.40 | 0.39 |
| Hybrid: TF-IDF + rule hits + LogReg | 0.28 | 0.57 |
| Rules (baseline) | 0.18 | 0.28 |

The chosen model finds every vulnerable test snippet (detection recall 1.00) but also flags
every safe one (detection precision 0.74). Test scores of the other models are reported for
transparency only; they did not influence the choice.

The dataset is synthetic: every snippet is generated from templates that reproduce
patterns from the OWASP LLM Top 10, framework documentation and public CVEs. No real
user data or third-party code is included.
