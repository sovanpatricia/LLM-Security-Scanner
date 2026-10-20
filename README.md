# LLM Security Scanner (MIRPR 2026-2027)

Detects OWASP Top 10 for LLM Applications (2025) risks in Python and TypeScript code
that integrates large language models. Lab 3 stage: an intelligent algorithm trained on
a small dataset.

## Layout

```
data/
  build_dataset.py     generates the development set (199 snippets, 59 templates)
  build_testset.py     generates the held-out test set (63 snippets, 35 templates)
  dataset.csv, testset.csv
src/llmscan/
  rules.py             34 hand-written regex rules (baseline)
  features.py          comment stripping, code tokenizer, rule-hit features
  models.py            the four learned models and their hyperparameter grids
  scan.py              command-line scanner that uses the trained model
experiments/
  run_experiments.py   EDA, template-disjoint split, tuning, test, CV, figures
reports/
  eda.json, results.json, figures/
models/
  scanner_model.joblib char n-gram Linear SVM trained on the development set
examples/
  vulnerable_app.py    small deliberately vulnerable app for the demo
```

## Run

```bash
pip install -r requirements.txt
python data/build_dataset.py        # optional: data/*.csv are already committed
python data/build_testset.py
python experiments/run_experiments.py   # ~20 s on a laptop CPU
PYTHONPATH=src python -m llmscan.scan examples/vulnerable_app.py
PYTHONPATH=src python -m llmscan.scan path/to/your/repo --format json
```

The scanner exits with code 1 when it reports findings, so it can fail a CI job.

## Lab 3 results (held-out test set, macro-F1 over 9 classes)

| Model | Grouped CV (dev) | Test |
| --- | --- | --- |
| Rules (baseline) | 0.98 | 0.18 |
| Hybrid: TF-IDF + rule hits + LogReg | 0.47 | 0.10 |
| TF-IDF char n-grams + Linear SVM | 0.30 | **0.64** |
| TF-IDF tokens + Logistic Regression | 0.25 | 0.60 |
| TF-IDF tokens + Random Forest | 0.26 | 0.37 |

The dataset is synthetic: every snippet is generated from templates that reproduce
patterns from the OWASP LLM Top 10, framework documentation and public CVEs. No real
user data or third-party code is included.
