"""Figure for step 4: test-set confusion matrices of the step-3 model and the
model chosen in step 4, side by side. Reads saved predictions only; nothing is
retrained or rescored.

Run from the repository root:  python experiments/plot_step4.py
Inputs:  reports/pilot/app_results.json, reports/pilot/final_analysis.json
Output:  reports/figures/confusion_app_test_step4.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'reports/figures/confusion_app_test_step4.png'
CLASSES = ['LLM02', 'LLM05', 'LLM06', 'SAFE']
SHOWN = CLASSES + ['other']


def matrix(predictions):
    rows = pd.DataFrame(predictions)
    rows['pred'] = rows.pred.where(rows.pred.isin(CLASSES), 'other')
    return pd.crosstab(rows.label, rows.pred).reindex(index=CLASSES, columns=SHOWN, fill_value=0).values


step3 = json.loads((ROOT / 'reports/pilot/app_results.json').read_text())
step4 = json.loads((ROOT / 'reports/pilot/final_analysis.json').read_text())
panels = [('Step 3: char SVM, whole files (macro-F1 %.2f)' % step3['final_test']['macro_f1'],
           matrix(step3['test_predictions'])),
          ('Step 4: char SVM, functions + real data (macro-F1 %.2f)' % step4['final_test']['macro_f1'],
           matrix(step4['test_predictions']))]
top = max(m.max() for _, m in panels)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
for ax, (title, m) in zip(axes, panels):
    ax.imshow(m, cmap='Blues', vmin=0, vmax=top)
    ax.set_xticks(range(len(SHOWN)), SHOWN)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(SHOWN)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha='center', va='center',
                        color='white' if m[i, j] > top / 2 else '#0b0b0b')
    ax.set_xlabel('predicted', color='#52514e')
    ax.set_ylabel('true', color='#52514e')
    ax.set_title(title, fontsize=9, loc='left')
fig.tight_layout()
fig.savefig(FIG, dpi=150, facecolor='white')
print('saved', FIG.relative_to(ROOT))
