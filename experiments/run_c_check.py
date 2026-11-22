"""Step 4 (part 3b): extend the SVM C grid below 0.03, the edge of the part-3 grid.

Run from the repository root:  python experiments/run_c_check.py
Output: reports/pilot/improvements_part3b.json

Same model, data, regimes and selection rule as run_tuning.py (validation
applications only); only C changes: {0.001, 0.003, 0.01, 0.03, 0.1}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from run_app_experiments import app_metrics, key  # noqa: E402
from run_improvements import aggregate, confidence, unit_table  # noqa: E402
from run_tuning import make  # noqa: E402

OUT = ROOT / 'reports/pilot/improvements_part3b.json'
CS = (0.001, 0.003, 0.01, 0.03, 0.1)


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    synth = synth[['code', 'label']].assign(is_real=False)
    app = pd.read_csv(ROOT / 'data/appset.csv')
    val = app[app.split == 'app_validation']
    vunits = unit_table(app).query("split == 'app_validation'")
    groups = sorted(val.group.unique())

    def evaluate(hp, weight):
        labels = pd.Series(index=vunits.index, dtype=object)
        conf = pd.Series(index=vunits.index, dtype=float)
        folds = [(None, vunits)] if weight is None else [(g, vunits[vunits.group == g]) for g in groups]
        for held, rows in folds:
            train = synth if held is None else pd.concat(
                [synth, vunits[vunits.group != held][['code', 'label']].assign(is_real=True)])
            w = np.where(train.is_real, float(weight or 1), 1.0)
            m = make('svm_ext', hp).fit(train.code, train.label, clf__sample_weight=w)
            labels.loc[rows.index] = m.predict(list(rows.code))
            conf.loc[rows.index] = confidence(m, list(rows.code))
        return app_metrics(val, aggregate(val, vunits, labels, conf))

    res = {'protocol': __doc__.strip(), 'per_C': {}}
    for regime, weights in (('synthetic_only', (None,)), ('with_real', (1, 5))):
        for c in CS:
            scored = [(evaluate({'C': c, 'ngram': n, 'class_weight': cw}, w),
                       {'C': c, 'ngram': n, 'class_weight': cw, 'real_weight': w})
                      for n in ((2, 4), (3, 5)) for cw in ('balanced', None) for w in weights]
            best, hp = max(scored, key=lambda s: key(s[0]))
            res['per_C'][f'{regime}/C={c}'] = {'best_hp': hp, 'val': best}
            print(f'{regime:15s} C={c:<6} macroF1={best["macro_f1"]:.3f} pairAcc={best["pair_accuracy"]:.2f} '
                  f'detP={best["detect_precision"]:.2f} detR={best["detect_recall"]:.2f} hp={hp}')
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
