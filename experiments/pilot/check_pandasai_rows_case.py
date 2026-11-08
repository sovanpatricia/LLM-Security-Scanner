"""Check the PandasAI rows-in-prompt case with dummy personal data.

Run from the repository root. PandasAI and a live LLM are not needed: a small
stand-in provides the schema metadata the serializer reads.
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import joblib
import pandas as pd
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASE = Path('data/pilot/cases/pandasai_rows_in_prompt')
SECRET = 'dummy.customer@example.com'


class DataFrameStandIn:
    """Provides what DataframeSerializer.serialize reads from a PandasAI DataFrame."""

    def __init__(self, data):
        self._df = data
        self.schema = SimpleNamespace(name='customers', description=None, columns=[])
        self.rows_count, self.columns_count = data.shape

    def head(self, n=5):
        return self._df.head(n)


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    data = pd.DataFrame({'name': ['Ana', 'Bob'], 'email': [SECRET, 'second@example.com'], 'spend': [120, 80]})
    result = {
        'case_id': 'pandasai_rows_in_prompt_001',
        'scope': 'Extracted serializer on a dataframe stand-in with dummy personal data; no live model call',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = module.DataframeSerializer.serialize(DataFrameStandIn(data), dialect='postgres')
        assert 'name,email,spend' in text and 'dimensions="2x3"' in text, (variant, 'schema must still be sent')
        leaked = SECRET in text
        assert leaked == (variant == 'vulnerable'), (variant, 'only the vulnerable version may send rows')
        code = path.read_text()
        result['variants'][variant] = {
            'serialized_prompt_part': text,
            'personal_data_in_prompt': leaked,
            'whole_file_model_prediction': str(model.predict([code])[0]),
            'whole_file_rules_prediction': rule_predict(code),
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
        }
    out = ROOT / 'reports/pilot/pandasai_rows_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
