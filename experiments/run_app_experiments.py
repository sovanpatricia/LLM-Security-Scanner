"""Step 3 experiments on real application code (data/appset.csv).

Run from the repository root:  python experiments/run_app_experiments.py
Outputs: reports/pilot/app_results.json, reports/figures/confusion_app_test.png

PROTOCOL (fixed before any score on the application test set was computed):
  Data. Synthetic = Lab 3 dataset.csv + val_newstyle.csv (262 snippets).
        app_validation = DVLA + streamlit-agent, app_test = Vanna + PandasAI
        (data/pilot/splits.json). Each app case gives a vulnerable file
        (OWASP label) and a control file (SAFE).
  A. Synthetic-only. Every model/hyperparameter setting in GRIDS is trained on
     the synthetic data and scored on app_validation. Score = macro-F1 over the
     classes present in the app data (LLM02, LLM05, LLM06, SAFE); ties broken by
     pair accuracy, then detection F1. Best setting per model; the chosen model
     is the best model overall (rules baseline included).
  B. Combined. Each model with its stage-A setting is trained on synthetic data
     plus ONE validation app and scored on the OTHER validation app (both ways,
     predictions pooled). The chosen model uses the combined regime only if this
     beats its stage-A validation score on the same rows.
  C. Test. The chosen model, in the chosen regime, is trained on synthetic data
     (+ both validation apps if combined) and scored ONCE on app_test. Test
     scores of all other models and regimes are reported for transparency only.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llmscan.features import strip_comments  # noqa: E402
from llmscan.models import GRIDS, PRETTY, build  # noqa: E402
from llmscan.rules import rule_predict  # noqa: E402

APP_CLASSES = ['LLM02', 'LLM05', 'LLM06', 'SAFE']
OUT = ROOT / 'reports/pilot/app_results.json'
FIG = ROOT / 'reports/figures/confusion_app_test.png'


def tuple_hp(hp):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}


def listed(hp):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}


def app_metrics(rows, pred):
    rows = rows.assign(pred=list(pred))
    y, p = rows.label, rows.pred
    yb, pb = (y != 'SAFE').astype(int), (p != 'SAFE').astype(int)
    pairs = []
    for _, g in rows.groupby('pair_id'):
        v = g[g.variant == 'vulnerable'].iloc[0]
        c = g[g.variant == 'control'].iloc[0]
        pairs.append({'separated': v.pred != 'SAFE' and c.pred == 'SAFE',
                      'exact': v.pred == v.label and c.pred == 'SAFE',
                      'differ': v.pred != c.pred})
    pairs = pd.DataFrame(pairs)
    return {
        'n_rows': len(rows),
        'accuracy': accuracy_score(y, p),
        'macro_f1': f1_score(y, p, labels=APP_CLASSES, average='macro', zero_division=0),
        'detect_precision': precision_score(yb, pb, zero_division=0),
        'detect_recall': recall_score(yb, pb, zero_division=0),
        'detect_f1': f1_score(yb, pb, zero_division=0),
        'pair_accuracy': float(pairs.exact.mean()),
        'pairs_separated': float(pairs.separated.mean()),
        'pairs_with_different_predictions': float(pairs.differ.mean()),
    }


def key(m):
    return (m['macro_f1'], m['pair_accuracy'], m['detect_f1'])


def rules(codes):
    return [rule_predict(strip_comments(c)) for c in codes]


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    app = pd.read_csv(ROOT / 'data/appset.csv')
    val, test = app[app.split == 'app_validation'], app[app.split == 'app_test']
    val_groups = sorted(val.group.unique())
    res = {'protocol': __doc__.split('PROTOCOL')[1].strip(),
           'sizes': {'synthetic_train': len(synth), 'app_validation': len(val), 'app_test': len(test)},
           'validation_groups': val_groups, 'test_groups': sorted(test.group.unique()),
           'models': {}}

    # ---- A. synthetic-only, tuned on app_validation
    res['models']['rules'] = {'best_hp': None, 'grid_size': 0, 'A_val': app_metrics(val, rules(val.code))}
    for name, grid in GRIDS.items():
        scored = []
        for hp in grid:
            m = build(name, **hp).fit(synth.code, synth.label)
            scored.append((app_metrics(val, m.predict(val.code)), hp))
        best, hp = max(scored, key=lambda s: key(s[0]))
        res['models'][name] = {'best_hp': listed(hp), 'grid_size': len(grid), 'A_val': best,
                               'A_val_all_settings': [{'hp': listed(h), 'macro_f1': s['macro_f1']} for s, h in scored]}
        print(f'A  {name:15s} val macroF1={best["macro_f1"]:.3f} pairAcc={best["pair_accuracy"]:.2f} hp={hp}')
    print(f'A  rules           val macroF1={res["models"]["rules"]["A_val"]["macro_f1"]:.3f}')
    chosen = max(res['models'], key=lambda k: key(res['models'][k]['A_val']))

    # ---- B. combined regime, leave-one-validation-app-out
    for name in GRIDS:
        hp = tuple_hp(res['models'][name]['best_hp'])
        pooled = []
        for held in val_groups:
            train = pd.concat([synth, val[val.group != held]])
            rows = val[val.group == held]
            pooled.append(rows.assign(pred=build(name, **hp).fit(train.code, train.label).predict(rows.code)))
        pooled = pd.concat(pooled).loc[val.index]
        res['models'][name]['B_val_leave_one_app_out'] = app_metrics(val, pooled.pred)
        print(f'B  {name:15s} val macroF1={res["models"][name]["B_val_leave_one_app_out"]["macro_f1"]:.3f}')
    regime = 'synthetic_only'
    if chosen != 'rules' and key(res['models'][chosen]['B_val_leave_one_app_out']) > key(res['models'][chosen]['A_val']):
        regime = 'combined'
    res['chosen'] = {'model': chosen, 'regime': regime, 'pretty': PRETTY[chosen]}
    print('CHOSEN ON VALIDATION:', chosen, regime)

    # ---- C. test (once for the chosen setting; others for transparency)
    preds = {}
    for name in res['models']:
        if name == 'rules':
            preds[('rules', 'synthetic_only')] = rules(test.code)
            res['models'][name]['test_synthetic_only'] = app_metrics(test, preds[('rules', 'synthetic_only')])
            continue
        hp = tuple_hp(res['models'][name]['best_hp'])
        for reg, train in (('synthetic_only', synth), ('combined', pd.concat([synth, val]))):
            p = list(build(name, **hp).fit(train.code, train.label).predict(test.code))
            preds[(name, reg)] = p
            res['models'][name][f'test_{reg}'] = app_metrics(test, p)
    final = preds[(chosen, regime)]
    res['final_test'] = res['models'][chosen][f'test_{regime}']
    print(f'TEST ({chosen}, {regime}): macroF1={res["final_test"]["macro_f1"]:.3f} '
          f'pairAcc={res["final_test"]["pair_accuracy"]:.2f} detectF1={res["final_test"]["detect_f1"]:.3f}')

    # ---- analysis of the final model on the test set
    rows = test.assign(pred=final)
    shown = APP_CLASSES + ['other']
    rows = rows.assign(pred_group=rows.pred.where(rows.pred.isin(APP_CLASSES), 'other'))
    cm = pd.crosstab(rows.label, rows.pred_group).reindex(index=APP_CLASSES, columns=shown, fill_value=0)
    res['confusion_test'] = {'rows_true': APP_CLASSES, 'cols_pred': shown, 'matrix': cm.values.tolist()}
    res['predicted_outside_app_classes'] = rows[rows.pred_group == 'other'].pred.value_counts().to_dict()
    res['per_class_test'] = classification_report(rows.label, rows.pred, labels=APP_CLASSES,
                                                  output_dict=True, zero_division=0)
    res['per_form_test'] = rows.groupby('form').apply(
        lambda g: float((g.label == g.pred).mean()), include_groups=False).to_dict()
    cols = ['id', 'group', 'form', 'label', 'pred']
    res['test_predictions'] = rows[cols].to_dict(orient='records')
    res['examples_correct'] = rows[rows.label == rows.pred][cols].to_dict(orient='records')
    res['examples_wrong'] = rows[rows.label != rows.pred][cols].to_dict(orient='records')

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(cm.values, cmap='Blues', vmin=0)
    ax.set_xticks(range(len(shown)), shown)
    ax.set_yticks(range(len(APP_CLASSES)), APP_CLASSES)
    for i in range(len(APP_CLASSES)):
        for j in range(len(shown)):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, v, ha='center', va='center', color='white' if v > cm.values.max() / 2 else '#0b0b0b')
    ax.set_xlabel('predicted', color='#52514e')
    ax.set_ylabel('true', color='#52514e')
    ax.set_title(f'App test set: {PRETTY[chosen]} ({regime.replace("_", "-")})', fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, facecolor='white')
    plt.close(fig)
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
