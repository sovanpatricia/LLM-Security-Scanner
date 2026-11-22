"""Step 4 (part 3): wider hyperparameter search, regularization and a small
neural network, on function-level units, validation applications only.

Run from the repository root:  python experiments/run_tuning.py
Output: reports/pilot/improvements_part3.json

Every setting is scored in two regimes, as in parts 1 and 2: synthetic training
only, and synthetic training plus the other validation application
(leave-one-app-out), real examples weighted 1 or 5.

Models and grids:
  svm_ext     char n-gram TF-IDF + Linear SVM; C in {0.03, 0.1, 0.3, 1, 3},
              n-grams 2-4 or 3-5, class weights balanced or none
  logreg_ext  token TF-IDF + Logistic Regression (saga); L2 or L1 penalty
              (L1 sets most weights to zero: a smaller, easier to read model),
              C in {0.3, 1, 3, 10}
  mlp         token TF-IDF + multilayer perceptron; hidden layers (64), (128) or
              (128, 64); optimizer Adam (learning rate 0.001) or SGD with momentum
              (learning rate 0.01); L2 penalty alpha in {1e-4, 1e-3, 1e-2};
              early stopping on 15% of the training data; max 300 epochs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from llmscan.features import code_tokens, strip_comments  # noqa: E402
from llmscan.models import SEED  # noqa: E402
from run_app_experiments import app_metrics, key  # noqa: E402
from run_improvements import aggregate, confidence, unit_table  # noqa: E402

OUT = ROOT / 'reports/pilot/improvements_part3.json'


def _strip(X):
    return [strip_comments(c) for c in X]


def tokens(ngram=(1, 2)):
    return TfidfVectorizer(tokenizer=code_tokens, token_pattern=None, lowercase=False,
                           ngram_range=ngram, sublinear_tf=True)


def make(model, hp):
    clean = ('strip_comments', FunctionTransformer(_strip))
    if model == 'svm_ext':
        feats = TfidfVectorizer(analyzer='char_wb', ngram_range=tuple(hp['ngram']), sublinear_tf=True)
        clf = LinearSVC(C=hp['C'], class_weight=hp['class_weight'], random_state=SEED)
    elif model == 'logreg_ext':
        feats = tokens()
        clf = LogisticRegression(C=hp['C'], l1_ratio=1.0 if hp['penalty'] == 'l1' else 0.0, solver='saga',
                                 max_iter=5000, class_weight='balanced', random_state=SEED)
    elif model == 'mlp':
        feats = tokens()
        sgd = hp['optimizer'] == 'sgd'
        clf = MLPClassifier(hidden_layer_sizes=tuple(hp['hidden']), solver=hp['optimizer'], alpha=hp['alpha'],
                            learning_rate_init=0.01 if sgd else 0.001, momentum=0.9,
                            early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=SEED)
    else:
        raise ValueError(model)
    return Pipeline([clean, ('features', feats), ('clf', clf)])


GRIDS = {
    'svm_ext': [{'C': c, 'ngram': n, 'class_weight': w} for c in (0.03, 0.1, 0.3, 1, 3)
                for n in ((2, 4), (3, 5)) for w in ('balanced', None)],
    'logreg_ext': [{'penalty': p, 'C': c} for p in ('l2', 'l1') for c in (0.3, 1, 3, 10)],
    'mlp': [{'hidden': h, 'optimizer': o, 'alpha': a} for h in ((64,), (128,), (128, 64))
            for o in ('adam', 'sgd') for a in (1e-4, 1e-3, 1e-2)],
}
REAL_WEIGHTS = (1, 5)


def main():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    synth = synth[['code', 'label']].assign(is_real=False)
    app = pd.read_csv(ROOT / 'data/appset.csv')
    val = app[app.split == 'app_validation']
    vunits = unit_table(app).query("split == 'app_validation'")
    groups = sorted(val.group.unique())

    def evaluate(model, hp, weight):
        labels = pd.Series(index=vunits.index, dtype=object)
        conf = pd.Series(index=vunits.index, dtype=float)
        folds = [(None, vunits)] if weight is None else [(g, vunits[vunits.group == g]) for g in groups]
        for held, rows in folds:
            train = synth if held is None else pd.concat(
                [synth, vunits[vunits.group != held][['code', 'label']].assign(is_real=True)])
            w = np.where(train.is_real, float(weight or 1), 1.0)
            m = make(model, hp).fit(train.code, train.label, clf__sample_weight=w)
            labels.loc[rows.index] = m.predict(list(rows.code))
            conf.loc[rows.index] = confidence(m, list(rows.code))
        return app_metrics(val, aggregate(val, vunits, labels, conf))

    res = {'protocol': __doc__.split('Models and grids:')[1].strip(), 'results': {}}
    for model, grid in GRIDS.items():
        for regime, weights in (('synthetic_only', (None,)), ('with_real', REAL_WEIGHTS)):
            scored = [(evaluate(model, hp, w), dict(hp, real_weight=w)) for hp in grid for w in weights]
            best, hp = max(scored, key=lambda s: key(s[0]))
            res['results'][f'{regime}/{model}'] = {'best_hp': hp, 'val': best, 'grid_size': len(scored),
                                                   'all': [{'hp': h, 'macro_f1': s['macro_f1'],
                                                            'pair_accuracy': s['pair_accuracy']} for s, h in scored]}
            print(f'{regime:15s} {model:11s} macroF1={best["macro_f1"]:.3f} pairAcc={best["pair_accuracy"]:.2f} '
                  f'detP={best["detect_precision"]:.2f} detR={best["detect_recall"]:.2f} hp={hp}')
    best = max(res['results'].items(), key=lambda kv: key(kv[1]['val']))
    res['best_on_validation'] = {'name': best[0], 'best_hp': best[1]['best_hp'], 'val': best[1]['val']}
    print('BEST ON VALIDATION:', best[0], round(best[1]['val']['macro_f1'], 3))
    OUT.write_text(json.dumps(res, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
