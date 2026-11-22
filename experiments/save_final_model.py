"""Train and save the model chosen in step 4, for the scanner (step 5).

Run from the repository root:  python experiments/save_final_model.py
Output: models/scanner_model_v2.joblib

The chosen model (reports/pilot/final_analysis.json, candidate svm_v2) is a
character 2-4-gram TF-IDF + Linear SVM with C=0.03 and balanced class weights,
trained on the synthetic snippets plus the function-level units of both
validation applications, real units weighted 5. It is built with
llmscan.models.build so that the saved file only depends on the llmscan
package, and it is trained with the scikit-learn version installed here, so it
loads without version warnings. No data from the test applications is used.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from llmscan.models import build  # noqa: E402
from run_improvements import unit_table  # noqa: E402

OUT = ROOT / 'models/scanner_model_v2.joblib'
REAL_WEIGHT = 5


def train():
    synth = pd.concat([pd.read_csv(ROOT / 'data/dataset.csv'), pd.read_csv(ROOT / 'data/val_newstyle.csv')])
    units = unit_table(pd.read_csv(ROOT / 'data/appset.csv'))
    real = units[units.split == 'app_validation']
    codes = list(synth.code) + list(real.code)
    labels = list(synth.label) + list(real.label)
    weights = np.r_[np.ones(len(synth)), np.full(len(real), float(REAL_WEIGHT))]
    model = build('char_svm', C=0.03, ngram=(2, 4))
    return model.fit(codes, labels, clf__sample_weight=weights), codes


if __name__ == '__main__':
    model, _ = train()
    joblib.dump(model, OUT)
    print('saved', OUT.relative_to(ROOT))
