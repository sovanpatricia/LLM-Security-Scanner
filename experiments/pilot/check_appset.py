"""Checks for data/appset.csv and the application split. Run from the repo root.

- every case's split field equals the split of its group in splits.json
- no application group, template family or pair crosses splits
- each split has every target class and SAFE
- appset.csv is up to date with the case folders
"""
import csv
import json
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = json.loads((ROOT / 'data/pilot/splits.json').read_text())['groups']
TARGETS = {'LLM02', 'LLM05', 'LLM06'}


def load_rows():
    with (ROOT / 'data/appset.csv').open() as handle:
        return list(csv.DictReader(handle))


class AppsetTests(unittest.TestCase):
    def test_appset_is_current(self):
        before = (ROOT / 'data/appset.csv').read_bytes()
        subprocess.run([sys.executable, str(ROOT / 'data/build_appset.py')], check=True, capture_output=True)
        self.assertEqual(before, (ROOT / 'data/appset.csv').read_bytes(), 'run data/build_appset.py and commit the result')

    def test_case_split_matches_group(self):
        for path in (ROOT / 'data/pilot/cases').glob('*/case.json'):
            case = json.loads(path.read_text())
            self.assertIn(case['group_id'], SPLITS, path)
            self.assertEqual(case['split'], SPLITS[case['group_id']], path)

    def test_no_group_family_or_pair_crosses_splits(self):
        for key in ('group', 'template', 'pair_id'):
            seen = defaultdict(set)
            for row in load_rows():
                seen[row[key]].add(row['split'])
            crossing = {k: v for k, v in seen.items() if len(v) > 1}
            self.assertEqual(crossing, {}, key)

    def test_every_split_covers_all_classes(self):
        labels = defaultdict(set)
        for row in load_rows():
            labels[row['split']].add(row['label'])
        self.assertEqual(set(labels), {'app_validation', 'app_test'})
        for split, found in labels.items():
            self.assertTrue(TARGETS | {'SAFE'} <= found, (split, found))

    def test_pairs_are_complete(self):
        pairs = defaultdict(list)
        for row in load_rows():
            pairs[row['pair_id']].append(row['label'])
        for pair, labels in pairs.items():
            self.assertEqual(len(labels), 2, pair)
            self.assertEqual(labels.count('SAFE'), 1, pair)


if __name__ == '__main__':
    unittest.main()
