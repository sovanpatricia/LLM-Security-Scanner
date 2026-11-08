"""Check the streamlit-agent read-only SQL case with simulated model-written SQL.

Run from the repository root. Needs SQLAlchemy; streamlit, LangChain and a live
LLM are not needed. SQLDatabase is replaced by a small stand-in that executes
statements inside engine.begin(), like langchain-community's SQLDatabase._execute.
"""
import ast
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
from sqlalchemy import create_engine, text
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASE = Path('data/pilot/cases/streamlit_agent_sql_readonly')
SELECT = 'SELECT Name FROM Artist ORDER BY Name'
DELETE = 'DELETE FROM Artist'  # what an injected or mistaken model could write


class SQLDatabaseStandIn:
    """Executes like langchain-community SQLDatabase: inside engine.begin()."""

    def __init__(self, engine):
        self._engine = engine

    def run(self, command):
        with self._engine.begin() as connection:
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                return [tuple(row) for row in cursor.fetchall()]
        return []


def load_configure_db(path, fake_file):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Import) and n.names[0].name == 'streamlit'
        or isinstance(n, ast.ImportFrom) and (n.module or '').startswith('langchain'))]
    stub_st = SimpleNamespace(cache_resource=lambda **kwargs: (lambda f: f))
    namespace = {'st': stub_st, 'SQLDatabase': SQLDatabaseStandIn, '__file__': str(fake_file)}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['configure_db'], namespace['LOCALDB']


def make_db(folder):
    db = Path(folder) / 'Chinook.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE Artist (Name TEXT)')
        conn.executemany('INSERT INTO Artist VALUES (?)', [('AC/DC',), ('Queen',)])
    return db


def count_rows(db):
    with sqlite3.connect(db) as conn:
        return conn.execute('SELECT COUNT(*) FROM Artist').fetchone()[0]


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {
        'case_id': 'streamlit_agent_sql_readonly_001',
        'scope': 'Simulated model-written SQL through a SQLDatabase stand-in; no live model or agent',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        with tempfile.TemporaryDirectory() as tmp:
            db_file = make_db(tmp)
            configure_db, localdb = load_configure_db(path, Path(tmp) / 'chat_with_sql_db.py')
            db = configure_db(localdb)
            names = [row[0] for row in db.run(SELECT)]
            assert names == ['AC/DC', 'Queen'], (variant, 'SELECT must work in both versions')
            try:
                db.run(DELETE)
                write_error = None
            except Exception as error:  # SQLAlchemy wraps the sqlite3 error
                write_error = str(error.__cause__ or error).splitlines()[0]
            remaining = count_rows(db_file)
            db._engine.dispose()
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
    out = ROOT / 'reports/pilot/streamlit_sql_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
