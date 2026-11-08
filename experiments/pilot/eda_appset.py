"""Step 3: short EDA of the application-based dataset (data/appset.csv),
compared with the Lab 3 synthetic training data. Run from the repository root.

Writes reports/pilot/eda_appset.json and reports/figures/eda_appset.png.
"""
import difflib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from llmscan.features import code_tokens, strip_comments

CLASSES = ['LLM02', 'LLM05', 'LLM06', 'SAFE']
COLORS = {'LLM02': '#2a78d6', 'LLM05': '#eb6834', 'LLM06': '#1baf7a', 'SAFE': '#eda100'}
SYNTH, APP = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def with_sizes(df):
    return df.assign(lines=df.code.str.count('\n') + 1,
                     tokens=df.code.map(lambda c: len(code_tokens(strip_comments(c)))))


def changed_lines(vulnerable, control):
    diff = difflib.unified_diff(vulnerable.splitlines(), control.splitlines(), lineterm='', n=0)
    return sum(1 for line in diff if line[:1] in '+-' and not line.startswith(('+++', '---')))


def describe(series):
    return {k: round(float(v), 1) for k, v in series.describe().items()}


def main():
    app = with_sizes(pd.read_csv('data/appset.csv'))
    synth = with_sizes(pd.concat([pd.read_csv('data/dataset.csv'), pd.read_csv('data/val_newstyle.csv')]))

    pairs = []
    for pair_id, group in app.groupby('pair_id'):
        vuln = group[group.variant == 'vulnerable'].iloc[0]
        ctrl = group[group.variant == 'control'].iloc[0]
        pairs.append({'pair_id': pair_id, 'label': vuln.label, 'split': vuln.split, 'form': vuln.form,
                      'group': vuln.group, 'changed_lines': changed_lines(vuln.code, ctrl.code),
                      'tokens': int(vuln.tokens)})
    pairs = pd.DataFrame(pairs)

    synth_vocab = set(t for c in synth.code for t in code_tokens(strip_comments(c)))
    app_tokens = [t for c in app.code for t in code_tokens(strip_comments(c))]
    unseen = sum(t not in synth_vocab for t in app_tokens) / len(app_tokens)

    out = {
        'rows': len(app), 'cases': int(app.pair_id.nunique()), 'apps': int(app.group.nunique()),
        'case_families': int(app.template.nunique()),
        'rows_per_split_and_class': pd.crosstab(app.split, app.label).to_dict(orient='index'),
        'cases_per_app_and_class': pd.crosstab(pairs.group, pairs.label).to_dict(orient='index'),
        'app_kind_cases': pairs.merge(app[['pair_id', 'app_kind']].drop_duplicates(), on='pair_id').app_kind.value_counts().to_dict(),
        'origin_of_vulnerable_file': app[app.variant == 'vulnerable'].origin.value_counts().to_dict(),
        'form_cases': pairs.form.value_counts().to_dict(),
        'review_status_cases': app.drop_duplicates('pair_id').review_status.value_counts().to_dict(),
        'tokens': {'app': describe(app.tokens), 'synthetic_training': describe(synth.tokens)},
        'lines': {'app': describe(app.lines), 'synthetic_training': describe(synth.lines)},
        'median_tokens_by_class_app': app.groupby('label').tokens.median().to_dict(),
        'changed_lines_per_pair': describe(pairs.changed_lines),
        'changed_lines_by_class': pairs.groupby('label').changed_lines.median().to_dict(),
        'share_of_app_tokens_unseen_in_synthetic_vocabulary': round(unseen, 3),
        'synthetic_training_classes': sorted(synth.label.unique()),
        'pairs': pairs.to_dict(orient='records'),
    }

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
        a.spines[['left', 'bottom']].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=9)
        a.set_axisbelow(True)

    table = pd.crosstab(app.split, app.label).reindex(['app_validation', 'app_test'])[CLASSES]
    bottom = [0, 0]
    for label in CLASSES:
        values = table[label].tolist()
        ax[0].bar(['validation', 'test'], values, bottom=bottom, width=0.5, color=COLORS[label],
                  edgecolor='white', linewidth=2, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax[0].set_title('Rows per split and class', color=INK, fontsize=11, loc='left')
    ax[0].set_ylabel('rows (files)', color=INK2)
    ax[0].yaxis.grid(True, color=GRID)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left')

    box = ax[1].boxplot([synth.tokens, app.tokens], tick_labels=['synthetic\n(training)', 'application\n(val + test)'],
                        widths=0.5, patch_artist=True, medianprops={'color': INK},
                        whiskerprops={'color': INK2}, capprops={'color': INK2})
    for patch, color in zip(box['boxes'], [SYNTH, APP]):
        patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(color)
    ax[1].set_yscale('log')
    ax[1].set_yticks([20, 50, 100, 200, 500])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title('Code tokens per file (log scale)', color=INK, fontsize=11, loc='left')
    ax[1].yaxis.grid(True, color=GRID, which='major')

    counts = pairs.changed_lines.value_counts().sort_index()
    ax[2].bar(counts.index.astype(str), counts.values, width=0.6, color=APP, edgecolor='white', linewidth=2)
    ax[2].set_title('Lines that differ inside a pair', color=INK, fontsize=11, loc='left')
    ax[2].set_xlabel('changed lines (vulnerable vs control)', color=INK2)
    ax[2].set_ylabel('pairs', color=INK2)
    ax[2].yaxis.grid(True, color=GRID)

    fig.tight_layout()
    fig.savefig(ROOT / 'reports/figures/eda_appset.png', dpi=150, facecolor='white')
    plt.close(fig)

    (ROOT / 'reports/pilot/eda_appset.json').write_text(json.dumps(out, indent=2, default=str) + '\n')
    summary = {k: v for k, v in out.items() if k != 'pairs'}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
