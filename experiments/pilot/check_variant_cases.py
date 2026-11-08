"""Run the base cases' behavioural tests on their refactored variants (_v2, _v3).

Run from the repository root. Each variant keeps the base case's attack and
functional checks: the attack or leak must happen only with vulnerable.py, and normal use
must work on both files. Reuses helpers from the base check scripts.
"""
import importlib.util
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import joblib
import pandas as pd
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASES = Path('data/pilot/cases')


def script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_dvla_sql(path):
    base = script('check_first_case')
    context = json.loads((path.parent / 'context.json').read_text())
    wrapper = next(s['code'] for s in context['segments'] if s['code'].startswith('def get_transactions('))
    module = base.load(path)
    normal = base.returned_users(module, '1', wrapper)
    attack = base.returned_users(module, "1' OR '1'='1", wrapper)
    assert normal == ['1'], 'normal lookup must work'
    return {'normal_users': normal, 'injected_users': attack}, attack == ['1', '2']


def check_vanna_plotly(path):
    base = script('check_vanna_case')
    vn = base.load(path).VannaBase()
    df = pd.DataFrame({'name': ['ana', 'bob', 'cid'], 'salary': [10, 20, 30]})
    normal = vn.get_plotly_figure(plotly_code=base.BENIGN, df=df, dark_mode=False)
    assert normal is not None and normal.data[0].type == 'bar', 'chart must still work'
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / 'marker.txt'
        vn.get_plotly_figure(plotly_code=base.malicious_code(marker), df=df, dark_mode=False)
        executed = marker.exists()
    return {'normal_chart_type': normal.data[0].type, 'injected_code_executed': executed}, executed


def check_pandasai_exec(path):
    base = script('check_pandasai_case')
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / 'marker.txt'
        with_sandbox = base.make_agent(path, base.FakeSandbox()).execute_code(base.generated_code(marker))
        assert with_sandbox['value'] == 'ran in sandbox' and not marker.exists(), 'sandbox path must work'
        try:
            base.make_agent(path, None).execute_code(base.generated_code(marker))
        except RuntimeError:
            pass
        executed = marker.exists()
    return {'sandbox_result': with_sandbox['value'], 'injected_code_executed_in_process': executed}, executed


def check_vanna_sql_write(path):
    base = script('check_vanna_sql_case')
    Runner = base.load_runner(path)
    with tempfile.TemporaryDirectory() as tmp:
        db = base.make_db(tmp)
        runner = Runner(database_path=str(db))
        names = base.run(runner, base.SELECT)['name'].tolist()
        assert names == ['ana', 'bob'], 'SELECT must work'
        try:
            base.run(runner, base.DELETE)
        except Exception:
            pass
        remaining = base.count_rows(db)
    return {'select_rows': names, 'rows_after_delete': remaining}, remaining == 0


def check_streamlit_sql(path):
    base = script('check_streamlit_sql_case')
    with tempfile.TemporaryDirectory() as tmp:
        db_file = base.make_db(tmp)
        configure_db, localdb = base.load_configure_db(path, Path(tmp) / 'chat_with_sql_db.py')
        db = configure_db(localdb)
        names = [row[0] for row in db.run(base.SELECT)]
        assert names == ['AC/DC', 'Queen'], 'SELECT must work'
        try:
            db.run(base.DELETE)
        except Exception:
            pass
        remaining = base.count_rows(db_file)
        db._engine.dispose()
    return {'select_rows': names, 'rows_after_delete': remaining}, remaining == 0


def check_dvla_sensitive(path):
    base = script('check_sensitive_case')
    observation = base.tool_observation(path, '')
    records = json.loads(observation)
    assert [{k: r[k] for k in ('userId', 'username')} for r in records] == [{'userId': 1, 'username': 'alice'}], 'tool must still work'
    exposed = base.DUMMY_PASSWORD in observation
    return {'returned_fields': sorted(records[0]), 'dummy_password_in_tool_observation': exposed}, exposed


def check_vanna_data(path):
    base = script('check_vanna_data_case')
    vn = base.make_vanna(path)
    vn.generate_sql(question='How many customers do we have?')
    leaked = any(base.SECRET in prompt for prompt in vn.prompts)
    return {'prompts_sent': len(vn.prompts), 'database_row_in_prompt': leaked}, leaked


def check_pandasai_rows(path):
    base = script('check_pandasai_rows_case')
    module = script_path(path)
    data = pd.DataFrame({'name': ['Ana', 'Bob'], 'email': [base.SECRET, 'second@example.com'], 'spend': [120, 80]})
    text = module.DataframeSerializer.serialize(base.DataFrameStandIn(data), dialect='postgres')
    assert 'name,email,spend' in text and 'dimensions="2x3"' in text, 'schema must still be sent'
    leaked = base.SECRET in text
    return {'personal_data_in_prompt': leaked}, leaked


def script_path(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAMILIES = {
    'dvla_sql_output': check_dvla_sql,
    'vanna_plotly_exec': check_vanna_plotly,
    'pandasai_exec_without_sandbox': check_pandasai_exec,
    'vanna_sql_write_access': check_vanna_sql_write,
    'streamlit_agent_sql_readonly': check_streamlit_sql,
    'dvla_sensitive_tool_result': check_dvla_sensitive,
    'vanna_llm_sees_data': check_vanna_data,
    'pandasai_rows_in_prompt': check_pandasai_rows,
}


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {'scope': 'Behavioural tests of refactored variants; no live model call',
              'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
              'cases': {}}
    for family, check in FAMILIES.items():
        for folder in sorted(CASES.glob(f'{family}_v*')):
            case = json.loads((folder / 'case.json').read_text())
            entry = {'form': case['form'], 'variants': {}}
            for variant in ('vulnerable', 'control'):
                path = folder / f'{variant}.py'
                observed, attack_worked = check(path)
                assert attack_worked == (variant == 'vulnerable'), (folder.name, variant, observed)
                code = path.read_text()
                entry['variants'][variant] = dict(observed, **{
                    'whole_file_model_prediction': str(model.predict([code])[0]),
                    'whole_file_rules_prediction': rule_predict(code),
                    'scanner_eligible_units': len(list(iter_units(path))),
                    'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
                })
            result['cases'][case['case_id']] = entry
    out = ROOT / 'reports/pilot/variant_cases.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {v: (d.get('whole_file_model_prediction'), d.get('whole_file_rules_prediction'),
                              [f['category'] for f in d['scanner_findings']])
                          for v, d in e['variants'].items()} for k, e in result['cases'].items()}, indent=1))


if __name__ == '__main__':
    main()
