"""Step 4 (part 2): ensembles, on function-level units, validation apps only.

Run from the repository root:  python experiments/run_ensembles.py
Output: reports/pilot/improvements_part2.json

Base models use the settings chosen in part 1 (improvements_part1.json,
unit_real), including the weight of real examples. Every method is scored in
two regimes, as in part 1: synthetic training only, and synthetic training
plus the other validation application (leave-one-app-out).

Methods:
  soft_vote        average of the class probabilities of Logistic Regression,
                   Linear SVM (softmax of its scores) and Random Forest
  bagging_logreg   25 Logistic Regressions, each on a bootstrap sample
  boosting_gb      gradient-boosted trees on code-token TF-IDF
  rules_gate       model label only if a rule fires on the unit or the model's
                   confidence is among the top k of its alarms; otherwise SAFE
  rules_first      rule label when a rule fires; otherwise the model label if its
                   confidence is among the top k of its alarms, else SAFE
                   (k = 100%, 75%, 50% or 25%, chosen on validation; confidences
                   are relative because they spread over 9 classes, 0.13-0.22)
The gates use the best single model of part 1 (char SVM) and the soft vote.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from llmscan.features import code_tokens, strip_comments  # noqa: E402
from llmscan.models import SEED, build  # noqa: E402
from llmscan.rules import rule_predict  # noqa: E402
from run_app_experiments import app_metrics, key, tuple_hp  # noqa: E402
from run_improvements import aggregate, unit_table  # noqa: E402

PART1 = json.loads((ROOT / 'reports/pilot/improvements_part1.json').read_text())['configs']['unit_real']
OUT = ROOT / 'reports/pilot/improvements_part2.json'
KEEP = (1.0, 0.75, 0.5, 0.25)  # share of the model's alarms kept, most confident first


def tokens():
    return TfidfVectorizer(tokenizer=code_tokens, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), sublinear_tf=True)


def strip_step():
    return ('strip_comments', FunctionTransformer(lambda X: [strip_comments(c) for c in X]))


def bagging():
    base = LogisticRegression(C=1, max_iter=5000, class_weight='balanced', random_state=SEED)
    return Pipeline([strip_step(), ('tfidf', tokens()),
                     ('clf', BaggingClassifier(base, n_estimators=25, random_state=SEED))])


def boosting():
    return Pipeline([strip_step(), ('tfidf', tokens()),
                     ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                                        random_state=SEED))])


def proba(model, codes):
    """Class probabilities; the SVM's scores go through a softmax."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(codes)
    s = model.decision_function(codes)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def base_fit(name, train, weight_real):
    hp = tuple_hp(PART1[name]['best_hp'])
    hp.pop('real_weight', None)
    w = np.where(train.is_real, float(weight_real), 1.0)
    return build(name, **hp).fit(train.code, train.label, clf__sample_weight=w)


def predict_method(method, train, codes):
    """Return (labels, confidences, classes, probabilities) for the units in codes."""
    if method in ('bagging_logreg', 'boosting_gb'):
        model = bagging() if method == 'bagging_logreg' else boosting()
        w = np.where(train.is_real, float(PART1['tfidf_logreg']['best_hp']['real_weight']), 1.0)
        model.fit(train.code, train.label, clf__sample_weight=w)
        p = proba(model, codes)
        classes = model.classes_
    elif method == 'char_svm':
        model = base_fit('char_svm', train, PART1['char_svm']['best_hp']['real_weight'])
        p, classes = proba(model, codes), model.classes_
    elif method == 'soft_vote':
        parts = [base_fit(n, train, PART1[n]['best_hp']['real_weight'])
                 for n in ('tfidf_logreg', 'char_svm', 'random_forest')]
        classes = parts[0].classes_
        p = np.mean([pd.DataFrame(proba(m, codes), columns=m.classes_)[classes].values for m in parts], axis=0)
    else:
        raise ValueError(method)
    return classes[p.argmax(axis=1)], p.max(axis=1)


def gated(labels, conf, codes, t, rules_first):
    out = []
    for label, c, code in zip(labels, conf, codes):
        rule = rule_predict(strip_comments(code))
        if rules_first and rule != 'SAFE':
            out.append(rule)
        elif label != 'SAFE' and (c >= t or (not rules_first and rule != 'SAFE')):
            out.append(label)
        else:
            out.append('SAFE')
    return out


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    synth = synth[['code', 'label']].assign(is_real=False)
    app = pd.read_csv(ROOT / 'data/appset.csv')
    val = app[app.split == 'app_validation']
    vunits = unit_table(app).query("split == 'app_validation'")
    groups = sorted(val.group.unique())

    def predictions(method, regime):
        """Unit labels and confidences for every validation unit."""
        labels = pd.Series(index=vunits.index, dtype=object)
        conf = pd.Series(index=vunits.index, dtype=float)
        folds = [(None, vunits)] if regime == 'synthetic_only' else [(g, vunits[vunits.group == g]) for g in groups]
        for held, rows in folds:
            train = synth if held is None else pd.concat(
                [synth, vunits[vunits.group != held][['code', 'label']].assign(is_real=True)])
            lab, c = predict_method(method, train, list(rows.code))
            labels.loc[rows.index], conf.loc[rows.index] = lab, c
        return labels, conf

    res = {'protocol': __doc__.strip(), 'results': {}}
    for regime in ('synthetic_only', 'with_real'):
        for method in ('soft_vote', 'bagging_logreg', 'boosting_gb', 'char_svm'):
            labels, conf = predictions(method, regime)
            entries = {} if method == 'char_svm' else {method: (None, list(labels))}
            if method in ('soft_vote', 'char_svm'):
                for rf in (False, True):
                    name = f"{'rules_first' if rf else 'rules_gate'}({method})"
                    risky = conf[labels != 'SAFE']
                    scored = []
                    for keep in KEEP:
                        t = float(np.quantile(risky, 1 - keep)) if len(risky) else 0.0
                        pred = gated(labels, conf, vunits.code, t, rf)
                        scored.append((app_metrics(val, aggregate(val, vunits, pred, conf)), keep))
                    best, t = max(scored, key=lambda s: key(s[0]))
                    entries[name] = (t, None, best)
            for name, entry in entries.items():
                if len(entry) == 2:
                    t, lab = entry
                    m = app_metrics(val, aggregate(val, vunits, lab, conf))
                else:
                    t, _, m = entry
                res['results'][f'{regime}/{name}'] = {'keep_share': t, 'val': m}
                print(f'{regime:15s} {name:28s} keep={t}  macroF1={m["macro_f1"]:.3f} pairAcc={m["pair_accuracy"]:.2f} '
                      f'detP={m["detect_precision"]:.2f} detR={m["detect_recall"]:.2f}')
    best = max(res['results'].items(), key=lambda kv: key(kv[1]['val']))
    res['best_on_validation'] = {'name': best[0], **best[1]}
    print('BEST ENSEMBLE ON VALIDATION:', best[0], round(best[1]['val']['macro_f1'], 3))
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
