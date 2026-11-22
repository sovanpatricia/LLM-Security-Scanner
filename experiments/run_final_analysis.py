"""Step 4 (part 4): compare the final candidates on three criteria, choose one
on validation, then score it ONCE on the test applications.

Run from the repository root:  python experiments/run_final_analysis.py
Outputs: reports/pilot/final_analysis.json, models/scanner_model_v2.joblib

Candidates (settings chosen in parts 1-3, on validation only):
  step3_svm     step-3 model: char 3-5-gram SVM, C=1, whole file, synthetic data
  svm_v2        char 2-4-gram SVM, C=0.03, balanced classes, function level,
                real examples weighted 5
  logreg_v2     token TF-IDF + Logistic Regression, L2, C=1, function level,
                real examples weighted 5
  mlp_v2        token TF-IDF + MLP (128 hidden units, Adam, alpha=1e-4, early
                stopping), function level, real examples weighted 1

Criteria:
  1. performance   validation metrics (leave-one-app-out for real data)
  2. size, speed   trained on synthetic data + both validation apps: number of
                   features and parameters, saved file size, training time,
                   prediction time per function
  3. explainability  LIME on each validation vulnerable function: share of
                   explanations whose top 5 words (pushing towards the true
                   class) include a word from the lines that make the code
                   vulnerable; plus whether the model is directly readable
Training data and evaluation: the app split in data/pilot/splits.json is
unchanged, and the test apps (Vanna, PandasAI) are never used for training or
for any choice. Unlike step 3, the function-level candidates are also trained
on functions from the validation apps (DVLA, streamlit-agent), weighted as
above. Validation scores therefore come from leave-one-app-out: to score one
validation app, the model is trained on the synthetic data plus the other
validation app only, so the app being scored is never seen in training. The
final models are trained on the synthetic data plus both validation apps.

Choice rule: highest validation macro-F1; a candidate within 0.05 of it is
preferred if it is more explainable, smaller and faster. The choice uses
validation results only and is fixed before any test prediction is made.

Test: one run, after the choice. In that run all candidates are scored on the
test apps: the chosen model's score is the final evaluation, and the other
candidates' scores are reported for transparency only and must not be used
to change the choice. (The step-3 model's test score was already reported in
step 3.)
"""
import io
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from llmscan.models import build  # noqa: E402
from run_app_experiments import app_metrics  # noqa: E402
from run_improvements import aggregate, changed_lines, confidence, unit_table  # noqa: E402
from run_tuning import make  # noqa: E402

OUT = ROOT / 'reports/pilot/final_analysis.json'
MODEL_OUT = ROOT / 'models/scanner_model_v2.joblib'

CANDIDATES = {
    'step3_svm': {'factory': lambda: build('char_svm', C=1, ngram=(3, 5)), 'level': 'file', 'weight': None},
    'svm_v2': {'factory': lambda: make('svm_ext', {'C': 0.03, 'ngram': (2, 4), 'class_weight': 'balanced'}),
               'level': 'unit', 'weight': 5},
    'logreg_v2': {'factory': lambda: make('logreg_ext', {'penalty': 'l2', 'C': 1}), 'level': 'unit', 'weight': 5},
    'mlp_v2': {'factory': lambda: make('mlp', {'hidden': (128,), 'optimizer': 'adam', 'alpha': 1e-4}),
               'level': 'unit', 'weight': 1},
}


def proba(model, codes):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(codes)
    s = model.decision_function(codes)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit(spec, synth, real_units):
    model = spec['factory']()
    if spec['weight'] is None:
        return model.fit(synth.code, synth.label)
    train = pd.concat([synth.assign(is_real=False), real_units[['code', 'label']].assign(is_real=True)])
    w = np.where(train.is_real, float(spec['weight']), 1.0)
    return model.fit(train.code, train.label, clf__sample_weight=w)


def predict_files(spec, model, files, units):
    if spec['level'] == 'file':
        return list(model.predict(files.code))
    u = units[units.row_id.isin(files.id)]
    return aggregate(files, u, model.predict(list(u.code)), confidence(model, list(u.code)))


def n_parameters(model):
    clf = model.named_steps['clf']
    if hasattr(clf, 'coefs_'):
        return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    return int(clf.coef_.size + clf.intercept_.size)


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    synth = synth[['code', 'label']]
    app = pd.read_csv(ROOT / 'data/appset.csv')
    units = unit_table(app)
    val, test = app[app.split == 'app_validation'], app[app.split == 'app_test']
    vunits, tunits = units[units.split == 'app_validation'], units[units.split == 'app_test']
    groups = sorted(val.group.unique())
    res = {'protocol': __doc__.split('Candidates')[1].strip(), 'candidates': {}}

    for name, spec in CANDIDATES.items():
        out = res['candidates'][name] = {'level': spec['level'], 'real_weight': spec['weight']}

        # 1. performance on validation
        if spec['weight'] is None:
            preds = predict_files(spec, fit(spec, synth, vunits.iloc[:0]), val, vunits)
        else:
            preds = pd.Series(index=val.index, dtype=object)
            for held in groups:
                model = fit(spec, synth, vunits[vunits.group != held])
                hf = val[val.group == held]
                preds.loc[hf.index] = predict_files(spec, model, hf, vunits)
            preds = list(preds.loc[val.index])
        out['val'] = app_metrics(val, preds)

        # 2. size and speed: final model, trained on synthetic + both validation apps
        start = time.perf_counter()
        final = fit(spec, synth, vunits if spec['weight'] is not None else vunits.iloc[:0])
        out['train_seconds'] = round(time.perf_counter() - start, 2)
        buffer = io.BytesIO()
        joblib.dump(final, buffer)
        out['model_kb'] = round(buffer.getbuffer().nbytes / 1024, 1)
        out['features'] = len(final.named_steps.get('features', final.named_steps.get('tfidf')).vocabulary_)
        out['parameters'] = n_parameters(final)
        codes = list(pd.concat([vunits.code, tunits.code]))
        start = time.perf_counter()
        for _ in range(5):
            final.predict(codes)
        out['ms_per_function'] = round((time.perf_counter() - start) / (5 * len(codes)) * 1000, 3)

        # 3. explainability with LIME on validation vulnerable functions
        explainer = LimeTextExplainer(class_names=list(final.classes_), split_expression=r'\W+', bow=True,
                                      random_state=42)
        hits, examples = [], []
        for pid, g in val.groupby('pair_id'):
            v = g[g.variant == 'vulnerable'].iloc[0]
            c = g[g.variant == 'control'].iloc[0]
            bad_lines = changed_lines(v.code, c.code)
            bad_words = {w for i, line in enumerate(v.code.splitlines(), 1) if i in bad_lines
                         for w in re.findall(r'\w+', line)}
            text = vunits[vunits.row_id == v.id].code.iloc[0]
            idx = list(final.classes_).index(v.label)
            exp = explainer.explain_instance(text, lambda xs: proba(final, list(xs)), labels=[idx],
                                             num_features=10, num_samples=500)
            top = [w for w, weight in exp.as_list(label=idx) if weight > 0][:5]
            hits.append(any(w in bad_words for w in top))
            examples.append({'pair_id': pid, 'label': v.label, 'top_words': top,
                             'words_on_changed_lines': sorted(bad_words)[:20], 'hit': hits[-1]})
        out['lime_hit_rate'] = round(float(np.mean(hits)), 2)
        out['lime_examples'] = examples
        out['directly_readable'] = 'linear: one weight per feature and class' if 'mlp' not in name else \
            'no: hidden layer weights have no direct meaning'
        out['_final'] = final
        print(f'{name:10s} val F1={out["val"]["macro_f1"]:.3f} pair={out["val"]["pair_accuracy"]:.2f} '
              f'size={out["model_kb"]}KB feats={out["features"]} params={out["parameters"]} '
              f'train={out["train_seconds"]}s pred={out["ms_per_function"]}ms LIME={out["lime_hit_rate"]}')

    # choice (validation only)
    best_f1 = max(c['val']['macro_f1'] for c in res['candidates'].values())
    close = {n: c for n, c in res['candidates'].items() if best_f1 - c['val']['macro_f1'] <= 0.05}
    chosen = min(close, key=lambda n: (-close[n]['lime_hit_rate'], close[n]['model_kb'], close[n]['ms_per_function'],
                                        -close[n]['val']['macro_f1']))
    res['chosen'] = chosen
    res['within_005_of_best'] = sorted(close)
    print('CHOSEN ON VALIDATION:', chosen, 'candidates within 0.05:', sorted(close))

    # test: one run, after the choice above. All candidates are scored; only the
    # chosen model's score is the final evaluation, the others are for transparency.
    for name, spec in CANDIDATES.items():
        cand = res['candidates'][name]
        final = cand.pop('_final')
        preds = predict_files(spec, final, test, tunits)
        cand['test'] = app_metrics(test, preds)
        if name == chosen:
            joblib.dump(final, MODEL_OUT)
            rows = test.assign(pred=preds)
            cols = ['id', 'group', 'form', 'label', 'pred']
            res['test_predictions'] = rows[cols].to_dict(orient='records')
            res['per_class_test'] = pd.crosstab(rows.label, rows.pred).to_dict(orient='index')
    res['final_test'] = res['candidates'][chosen]['test']
    t = res['final_test']
    print(f'TEST ({chosen}): macroF1={t["macro_f1"]:.3f} pairAcc={t["pair_accuracy"]:.2f} '
          f'detP={t["detect_precision"]:.2f} detR={t["detect_recall"]:.2f}')
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
