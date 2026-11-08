"""Check the Vanna plotly-exec case with simulated model output. Run from repo root.

No Vanna installation, database or live LLM is needed: the harness passes a
benign and a malicious plotly_code string to the extracted get_plotly_figure.
"""
import importlib.util
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import joblib
import pandas as pd
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASE = Path('data/pilot/cases/vanna_plotly_exec')
BENIGN = "fig = px.bar(df, x='name', y='salary')"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def malicious_code(marker):
    # What a prompt-injected model could return: a valid chart plus a side effect.
    # The side effect only writes a harmless marker file in a temporary folder.
    return (f"open({str(marker)!r}, 'w').write('executed')\n" + BENIGN)


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    df = pd.DataFrame({'name': ['ana', 'bob', 'cid'], 'salary': [10, 20, 30]})
    result = {
        'case_id': 'vanna_plotly_exec_001',
        'scope': 'Simulated model output passed to the extracted get_plotly_figure; no live model call',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        vn = load(path).VannaBase()
        normal = vn.get_plotly_figure(plotly_code=BENIGN, df=df, dark_mode=False)
        assert normal is not None and normal.data[0].type == 'bar', (variant, 'chart must still work')
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'marker.txt'
            attacked = vn.get_plotly_figure(plotly_code=malicious_code(marker), df=df, dark_mode=False)
            executed = marker.exists()
        assert attacked is not None, (variant, 'chart must still be returned')
        assert executed == (variant == 'vulnerable'), (variant, 'only the vulnerable version may run model code')
        code = path.read_text()
        result['variants'][variant] = {
            'normal_chart_type': normal.data[0].type,
            'injected_code_executed': executed,
            'whole_file_model_prediction': str(model.predict([code])[0]),
            'whole_file_rules_prediction': rule_predict(code),
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
        }
    out = ROOT / 'reports/pilot/vanna_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
