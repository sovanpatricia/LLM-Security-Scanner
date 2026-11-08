"""Build data/appset.csv from the application-based pilot cases.

Each case folder gives two rows: vulnerable.py (label = the case's OWASP
category) and control.py (label = SAFE, meaning protected against the case's
target vulnerability, not certified safe in general). The split of every row
comes from data/pilot/splits.json, by application group. Columns match the
Lab 3 CSVs (id, template, language, label, vulnerable, code) plus metadata.

Run from the repository root:  python data/build_appset.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'data/pilot/cases'
SPLITS = json.loads((ROOT / 'data/pilot/splits.json').read_text())['groups']
OUT = ROOT / 'data/appset.csv'
COLUMNS = ['id', 'template', 'language', 'label', 'vulnerable', 'code',
           'case_id', 'pair_id', 'group', 'split', 'variant', 'origin', 'form',
           'app_kind', 'review_status']


def family(case):
    """Base case and its refactored variants share one template id."""
    return case.get('variant_of', case['case_id']).rsplit('_', 1)[0]


def rows():
    for path in sorted(CASES.glob('*/case.json')):
        case = json.loads(path.read_text())
        split = SPLITS[case['group_id']]
        for variant in ('vulnerable', 'control'):
            entry = case['variants'][variant]
            yield {
                'id': f"{case['case_id']}:{variant}",
                'template': family(case),
                'language': 'python',
                'label': case['label']['owasp'] if variant == 'vulnerable' else 'SAFE',
                'vulnerable': int(variant == 'vulnerable'),
                'code': (path.parent / entry['file']).read_text(),
                'case_id': case['case_id'],
                'pair_id': case['case_id'],
                'group': case['group_id'],
                'split': split,
                'variant': variant,
                'origin': entry['origin'],
                'form': case.get('form', 'base'),
                'app_kind': case['source']['app_kind'],
                'review_status': case['review']['status'],
            }


def main():
    data = list(rows())
    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(data)
    counts = {}
    for row in data:
        key = (row['split'], row['label'])
        counts[key] = counts.get(key, 0) + 1
    print(f'{len(data)} rows written to {OUT.relative_to(ROOT)}')
    for (split, label), n in sorted(counts.items()):
        print(f'  {split:15} {label:6} {n}')


if __name__ == '__main__':
    main()
