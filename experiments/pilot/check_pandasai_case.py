"""Check the PandasAI exec-without-sandbox case with simulated generated code.

Run from the repository root. PandasAI and a live LLM are not needed: the
upstream CodeExecutor excerpt in context.json is loaded with small stubs.
"""
import ast
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import joblib
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASE = Path('data/pilot/cases/pandasai_exec_without_sandbox')


class CodeExecutionError(Exception):
    pass


class NoResultFoundError(Exception):
    pass


def upstream_code_executor():
    context = json.loads((CASE / 'context.json').read_text())
    code = next(s['code'] for s in context['segments'] if s['code'].startswith('class CodeExecutor'))
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.ClassDef)
    namespace = {'Any': object, 'Config': object, 'get_environment': dict,
                 'CodeExecutionError': CodeExecutionError, 'NoResultFoundError': NoResultFoundError}
    exec(compile(tree, '<upstream CodeExecutor excerpt>', 'exec'), namespace)
    return namespace['CodeExecutor']


class FakeSandbox:
    """Receives the code but never runs it, standing in for an isolated sandbox."""

    def __init__(self):
        self.received = []

    def execute(self, code, environment):
        self.received.append(code)
        return {'type': 'string', 'value': 'ran in sandbox'}


def make_agent(path, sandbox):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and (n.module or '').startswith('pandasai'))]
    namespace = {'CodeExecutor': upstream_code_executor(), 'Sandbox': object}
    exec(compile(tree, str(path), 'exec'), namespace)
    agent = namespace['Agent'].__new__(namespace['Agent'])
    agent._state = SimpleNamespace(logger=SimpleNamespace(log=lambda message: None), config=None, skills=[])
    agent._sandbox = sandbox
    agent._execute_sql_query = lambda query: None
    return agent


def generated_code(marker):
    # What a prompt-injected model could return: a normal answer plus a side effect.
    # The side effect only writes a harmless marker file in a temporary folder.
    return (f"open({str(marker)!r}, 'w').write('executed')\n"
            "result = {'type': 'string', 'value': 'ok'}")


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {
        'case_id': 'pandasai_exec_without_sandbox_001',
        'scope': 'Simulated generated code passed to the extracted Agent.execute_code; no live model call',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'marker.txt'
            sandbox = FakeSandbox()
            with_sandbox = make_agent(path, sandbox).execute_code(generated_code(marker))
            assert with_sandbox['value'] == 'ran in sandbox' and not marker.exists(), (variant, 'sandbox path must work')
            try:
                make_agent(path, None).execute_code(generated_code(marker))
                refused = None
            except RuntimeError as error:
                refused = str(error)
            executed = marker.exists()
        assert executed == (variant == 'vulnerable'), (variant, 'only the vulnerable version may run code in-process')
        code = path.read_text()
        result['variants'][variant] = {
            'sandbox_result': with_sandbox['value'],
            'no_sandbox_refusal': refused,
            'injected_code_executed_in_process': executed,
            'whole_file_model_prediction': str(model.predict([code])[0]),
            'whole_file_rules_prediction': rule_predict(code),
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
        }
    out = ROOT / 'reports/pilot/pandasai_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
