"""Check a controlled LLM tool-result disclosure with dummy SQLite data.

Run from the repository root. No LangChain installation or live LLM is needed.
"""
import ast
import json
import platform
import sqlite3
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import sklearn
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan
from llmscan.context import extract_context

CASE = ROOT / 'data/pilot/cases/dvla_sensitive_tool_result'
DUMMY_PASSWORD = 'dummy-password-user-1'

class ToolStub:
    """Only replace tool dispatch; this is not a LangChain AgentExecutor."""
    def __init__(self, name, func, description):
        self.name, self.func, self.description = name, func, description

    def run(self, argument):
        return self.func(argument)

def tool_observation(path, argument):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module == 'langchain.agents'
    )]
    namespace = {'Tool': ToolStub}
    exec(compile(tree, str(path), 'exec'), namespace)
    extracted_db = namespace['TransactionDb']
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE Users (userId INTEGER, username TEXT, password TEXT)')
        conn.executemany('INSERT INTO Users VALUES (?, ?, ?)', [
            (1, 'alice', DUMMY_PASSWORD), (2, 'bob', 'dummy-password-user-2')
        ])
        class TestDb(extracted_db):
            def __init__(self):
                self.conn = conn

            def close(self):
                pass
        namespace['TransactionDb'] = TestDb
        tool = namespace['get_current_user_tool']
        assert tool.name == 'GetCurrentUser'
        return tool.run(argument)

def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {
        'case_id': 'dvla_sensitive_tool_result_001',
        'scope': 'Simulated read-only tool dispatch; actual SQLite and source-derived wrapper; no live model call',
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__},
        'model_load_warnings': sorted({str(w.message) for w in captured}),
        'variants': {}
    }
    files = {v: CASE / f'{v}.py' for v in ('control', 'vulnerable')}
    assert files['vulnerable'].read_text() == files['control'].read_text().replace(
        'SELECT userId,username FROM Users', 'SELECT userId,username,password FROM Users'
    ), 'Pair must differ only in the selected password field.'
    for variant, path in files.items():
        observation = tool_observation(path, '')
        records = json.loads(observation)
        assert [{k: r[k] for k in ('userId', 'username')} for r in records] == [
            {'userId': 1, 'username': 'alice'}
        ], 'Both versions must preserve legitimate functionality.'
        assert tool_observation(path, "2 OR 1=1") == observation, 'Tool argument must not change the fixed user.'
        exposed = DUMMY_PASSWORD in observation
        assert exposed == (variant == 'vulnerable'), 'Credential disclosure must differ between variants.'
        assert set(records[0]) == ({'userId', 'username', 'password'} if exposed else {'userId', 'username'})
        context = extract_context(path.read_text(), 'get_current_user_tool')
        context_dir = ROOT / 'reports/pilot/contexts'
        context_dir.mkdir(parents=True, exist_ok=True)
        context_file = context_dir / f'sensitive_{variant}.json'
        context_file.write_text(json.dumps(context.to_dict(), indent=2) + '\n')
        result['variants'][variant] = {
            'functional_check_passed': True,
            'fixed_user_check_passed': True,
            'dummy_password_in_tool_observation': exposed,
            'returned_fields': sorted(records[0]),
            'whole_case_model_prediction': str(model.predict([path.read_text()])[0]),
            'whole_case_rules_prediction': rule_predict(path.read_text()),
            'extracted_context': {
                'policy': context.policy,
                'root_symbol': context.root_symbol,
                'spans': context.spans,
                'external_imports': context.external_imports,
                'lines': len(context.code.splitlines()),
                'model_prediction': str(model.predict([context.code])[0]),
                'rules_prediction': rule_predict(context.code),
                'artifact': str(context_file.relative_to(ROOT)),
            },
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': scan(path)
        }
    out = ROOT / 'reports/pilot/sensitive_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
