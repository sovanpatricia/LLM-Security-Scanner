"""Check the Vanna SQLite write-access case with simulated model-written SQL.

Run from the repository root. No Vanna installation or live LLM is needed:
Vanna imports are replaced by stubs and a temporary SQLite database is used.
"""
import ast
import asyncio
import json
import os
import sqlite3
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

CASE = Path('data/pilot/cases/vanna_sql_write_access')
SELECT = 'SELECT name FROM customers ORDER BY name'
DELETE = 'DELETE FROM customers'  # what an injected or mistaken model could write


def load_runner(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and (n.module or '').startswith('vanna'))]
    namespace = {'SqlRunner': object, 'RunSqlToolArgs': object, 'ToolContext': object}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['SqliteRunner']


def make_db(folder):
    db = Path(folder) / 'shop.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE customers (name TEXT)')
        conn.executemany('INSERT INTO customers VALUES (?)', [('ana',), ('bob',)])
    return db


def count_rows(db):
    with sqlite3.connect(db) as conn:
        return conn.execute('SELECT COUNT(*) FROM customers').fetchone()[0]


def run(runner, sql):
    return asyncio.run(runner.run_sql(SimpleNamespace(sql=sql), context=None))


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {
        'case_id': 'vanna_sql_write_access_001',
        'scope': 'Simulated model-written SQL passed to the extracted SqliteRunner; no live model call',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        Runner = load_runner(path)
        with tempfile.TemporaryDirectory() as tmp:
            db = make_db(tmp)
            runner = Runner(database_path=str(db))
            names = run(runner, SELECT)['name'].tolist()
            assert names == ['ana', 'bob'], (variant, 'SELECT must work in both versions')
            try:
                run(runner, DELETE)
                write_error = None
            except sqlite3.OperationalError as error:
                write_error = str(error)
            remaining = count_rows(db)
        assert (remaining == 0) == (variant == 'vulnerable'), (variant, 'only the vulnerable version may delete')
        code = path.read_text()
        result['variants'][variant] = {
            'select_rows': names,
            'delete_error': write_error,
            'rows_after_delete': remaining,
            'whole_file_model_prediction': str(model.predict([code])[0]),
            'whole_file_rules_prediction': rule_predict(code),
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
        }
    out = ROOT / 'reports/pilot/vanna_sql_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
