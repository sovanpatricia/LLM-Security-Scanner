"""Validate one draft case and measure the unchanged scanner. Run from repo root."""
import importlib.util
import ast
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, python_units, scan

CASE = ROOT / 'data/pilot/cases/dvla_sql_output'

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def returned_users(module, value, wrapper_code):
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE Transactions (transactionId INTEGER, userId TEXT, amount REAL)')
        conn.executemany('INSERT INTO Transactions VALUES (?, ?, ?)', [(1, '1', 10), (2, '2', 20)])
        class TestDb:
            def get_user_transactions(self, user_id):
                return module.get_user_transactions(SimpleNamespace(conn=conn), user_id)

            def close(self):
                pass  # The enclosing context owns this in-memory connection.

        namespace = {'TransactionDb': TestDb}
        exec(compile(wrapper_code, '<upstream get_transactions excerpt>', 'exec'), namespace)
        # Simulated agent output, not an argument produced by a live model.
        tool_call = {'name': 'GetUserTransactions', 'arguments': value}
        tool_registry = {'GetUserTransactions': namespace['get_transactions']}
        result = tool_registry[tool_call['name']](tool_call['arguments'])
        return [row['userId'] for row in json.loads(result)]

def main():
    model = joblib.load(ROOT / 'models/scanner_model.joblib')
    context = json.loads((CASE / 'context.json').read_text())
    wrapper_code = next(s['code'] for s in context['segments'] if s['code'].startswith('def get_transactions('))
    # Validate the checked-in excerpt before executing only this local wrapper.
    tree = ast.parse(wrapper_code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    context_code = '\n\n'.join(s['code'] for s in context['segments'])
    result = {'case_id': 'dvla_sql_output_001', 'scope': 'Simulated agent tool call through upstream wrapper to SQLite; not a live agent attack', 'variants': {}}
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        module = load(path)
        normal = returned_users(module, '1', wrapper_code)
        attack = returned_users(module, "1' OR '1'='1", wrapper_code)
        assert normal == ['1'], (variant, normal)
        assert attack == (['1', '2'] if variant == 'vulnerable' else []), (variant, attack)
        units = python_units(path, path.read_text())
        code = next(code for _, code in units if code.startswith('def get_user_transactions'))
        eligible = list(iter_units(path))
        result['variants'][variant] = {
            'normal_users': normal, 'injected_users': attack,
            'direct_function_model_prediction': str(model.predict([code])[0]),
            'direct_function_rules_prediction': rule_predict(code),
            'function_plus_context_prediction': str(model.predict([context_code + '\n\n' + code])[0]),
            'scanner_eligible_units': len(eligible),
            'scanner_findings': scan(path),
        }
    out = ROOT / 'reports/pilot/first_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
