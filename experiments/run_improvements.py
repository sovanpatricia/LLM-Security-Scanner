"""Step 4 (part 1): function-level units and real training data.

Run from the repository root:  python experiments/run_improvements.py
Output: reports/pilot/improvements_part1.json

Selection uses validation applications only; the test applications are not
scored here. They are scored once, after all step-4 improvements are chosen.

Configurations (all use the Lab 3 model grids):
  file_synth    step-3 setting: one prediction per whole file, synthetic training
  unit_synth    improvement 1: one prediction per function/method (target units),
                synthetic training
  unit_real     improvements 1+2: as unit_synth, plus the other validation
                application's target units in training (leave-one-app-out),
                each weighted 1, 5 or 20 times a synthetic snippet

Target units of a pair are the functions or methods that contain the lines
where the vulnerable and control files differ. When the changed unit is a
constant, the units that use it are taken instead, with the constant
prepended as context. A file's prediction is SAFE when all its target units
are SAFE; otherwise the non-SAFE label with the highest score.
"""
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from llmscan.features import strip_comments  # noqa: E402
from llmscan.models import GRIDS, build  # noqa: E402
from llmscan.rules import rule_predict  # noqa: E402
from llmscan.units import function_units  # noqa: E402
from run_app_experiments import app_metrics, key, listed  # noqa: E402

OUT = ROOT / 'reports/pilot/improvements_part1.json'
REAL_WEIGHTS = (1, 5, 20)  # how much one real example counts against one synthetic snippet


def changed_lines(code, other):
    sm = difflib.SequenceMatcher(None, code.splitlines(), other.splitlines())
    return {i + 1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag != 'equal' for i in range(i1, max(i2, i1 + 1))}


def target_units(code, other):
    """Code strings of the units that hold this file's side of the pair difference."""
    units = function_units(code)
    changed = changed_lines(code, other)
    hit = [u for u in units if any(u[0] <= line <= u[1] for line in changed)]
    targets = []
    for start, end, kind, name, text in hit:
        if kind in ('assign', 'class_assign'):
            names = set(re.findall(r'^\s*([A-Za-z_]\w*)\s*[:=]', text, re.M))
            users = [u for u in units if u[4] != text and any(re.search(rf'\b{n}\b', u[4]) for n in names)]
            targets += [text + '\n\n' + u[4] for u in users] or [text]
        else:
            targets.append(text)
    return targets


def unit_table(app):
    rows = []
    for pid, g in app.groupby('pair_id'):
        side = {r.variant: r for r in g.itertuples()}
        for variant, other in (('vulnerable', 'control'), ('control', 'vulnerable')):
            r = side[variant]
            for i, text in enumerate(target_units(r.code, side[other].code)):
                rows.append({'pair_id': pid, 'variant': variant, 'label': r.label, 'group': r.group,
                             'split': r.split, 'row_id': r.id, 'unit': i, 'code': text})
    return pd.DataFrame(rows)


def confidence(model, codes):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(codes).max(axis=1)
    s = np.sort(model.decision_function(codes), axis=1)
    return s[:, -1] - s[:, -2]


def aggregate(files, units, pred, conf):
    """One label per file from its target units."""
    units = units.assign(pred=list(pred), conf=list(conf))
    out = []
    for row_id in files.id:
        u = units[units.row_id == row_id]
        risky = u[u.pred != 'SAFE']
        out.append('SAFE' if risky.empty else risky.sort_values('conf').iloc[-1].pred)
    return out


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    app = pd.read_csv(ROOT / 'data/appset.csv')
    val = app[app.split == 'app_validation']
    units = unit_table(app)
    vunits = units[units.split == 'app_validation']
    val_groups = sorted(val.group.unique())
    res = {'protocol': __doc__.split('Configurations')[1].strip(),
           'units': {'validation_target_units': len(vunits),
                     'test_target_units': int((units.split == 'app_test').sum()),
                     'median_lines_validation_units': float(vunits.code.str.count('\n').median() + 1),
                     'median_lines_validation_files': float(val.code.str.count('\n').median() + 1)},
           'configs': {}}

    def run(name, model_name, grid, fit_predict):
        scored = []
        for hp in grid:
            preds = fit_predict(model_name, hp)
            scored.append((app_metrics(val, preds), hp))
        best, hp = max(scored, key=lambda s: key(s[0]))
        res['configs'].setdefault(name, {})[model_name] = {'best_hp': listed(hp) if hp else None, 'val': best}
        print(f'{name:12s} {model_name:15s} macroF1={best["macro_f1"]:.3f} pairAcc={best["pair_accuracy"]:.2f} '
              f'detP={best["detect_precision"]:.2f} detR={best["detect_recall"]:.2f}')

    # rules, file vs unit level
    res['configs']['file_synth'] = {'rules': {'best_hp': None, 'val': app_metrics(
        val, [rule_predict(strip_comments(c)) for c in val.code])}}
    upred = [rule_predict(strip_comments(c)) for c in vunits.code]
    res['configs']['unit_synth'] = {'rules': {'best_hp': None, 'val': app_metrics(
        val, aggregate(val, vunits, upred, [1.0] * len(upred)))}}

    for model_name, grid in GRIDS.items():
        def file_synth(m, hp):
            return build(m, **hp).fit(synth.code, synth.label).predict(val.code)

        def unit_synth(m, hp):
            model = build(m, **hp).fit(synth.code, synth.label)
            return aggregate(val, vunits, model.predict(vunits.code), confidence(model, list(vunits.code)))

        def unit_real(m, hp):
            hp = dict(hp)
            weight = hp.pop('real_weight')
            preds = pd.Series(index=val.index, dtype=object)
            for held in val_groups:
                train_units = vunits[vunits.group != held]
                train = pd.concat([synth[['code', 'label']], train_units[['code', 'label']]])
                w = np.r_[np.ones(len(synth)), np.full(len(train_units), float(weight))]
                model = build(m, **hp).fit(train.code, train.label, clf__sample_weight=w)
                hu, hf = vunits[vunits.group == held], val[val.group == held]
                preds.loc[hf.index] = aggregate(hf, hu, model.predict(hu.code), confidence(model, list(hu.code)))
            return list(preds.loc[val.index])

        run('file_synth', model_name, grid, file_synth)
        run('unit_synth', model_name, grid, unit_synth)
        run('unit_real', model_name, [dict(hp, real_weight=w) for hp in grid for w in REAL_WEIGHTS], unit_real)

    flat = [(cfg, m, v['val']) for cfg, models in res['configs'].items() for m, v in models.items()]
    cfg, m, best = max(flat, key=lambda t: key(t[2]))
    res['best_on_validation'] = {'config': cfg, 'model': m, 'val': best}
    print('BEST ON VALIDATION:', cfg, m, round(best['macro_f1'], 3))
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
