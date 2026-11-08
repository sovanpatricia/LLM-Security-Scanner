"""Check the Vanna 'LLM sees data' case with a stubbed model and database.

Run from the repository root. No Vanna installation or live LLM is needed.
generate_sql is called with its default value, as the legacy Flask app's
fix-SQL endpoint does (see context.json).
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import joblib
from llmscan.rules import rule_predict
from llmscan.scan import iter_units, scan

CASE = Path('data/pilot/cases/vanna_llm_sees_data')
SECRET = 'dummy.customer@example.com'


class Rows:
    """Stands in for the DataFrame returned by run_sql (only to_markdown is used)."""

    def to_markdown(self):
        return f'| email |\n|---|\n| {SECRET} |'


def make_vanna(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubbedVanna(module.VannaBase):
        config = None

        def __init__(self):
            self.prompts = []

        def get_similar_question_sql(self, question, **kwargs):
            return []

        def get_related_ddl(self, question, **kwargs):
            return ['CREATE TABLE customers (email TEXT)']

        def get_related_documentation(self, question, **kwargs):
            return []

        def get_sql_prompt(self, initial_prompt, question, question_sql_list, ddl_list, doc_list, **kwargs):
            return '\n'.join([question] + ddl_list + doc_list)

        def submit_prompt(self, prompt, **kwargs):
            self.prompts.append(prompt)
            # First reply asks to look at the data, as the model may legitimately do.
            if len(self.prompts) == 1:
                return 'intermediate_sql\nSELECT email FROM customers LIMIT 5'
            return 'SELECT COUNT(*) FROM customers'

        def extract_sql(self, llm_response):
            return llm_response.splitlines()[-1]

        def run_sql(self, sql):
            return Rows()

        def log(self, message, title='Info'):
            pass

    return StubbedVanna()


def main():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model = joblib.load(ROOT / 'models/scanner_model.joblib')
    result = {
        'case_id': 'vanna_llm_sees_data_001',
        'scope': 'Stubbed retrieval, database and model; generate_sql called with its default value',
        'model_load_warnings': sorted({str(w.message).split('.')[0] for w in captured}),
        'variants': {},
    }
    for variant in ('vulnerable', 'control'):
        path = CASE / f'{variant}.py'
        vn = make_vanna(path)
        answer = vn.generate_sql(question='How many customers do we have?')
        leaked = any(SECRET in prompt for prompt in vn.prompts)
        assert leaked == (variant == 'vulnerable'), (variant, 'only the vulnerable version may send rows to the model')
        code = path.read_text()
        result['variants'][variant] = {
            'prompts_sent': len(vn.prompts),
            'database_row_in_prompt': leaked,
            'returned': answer[:80],
            'whole_file_model_prediction': str(model.predict([code])[0]),
            'whole_file_rules_prediction': rule_predict(code),
            'scanner_eligible_units': len(list(iter_units(path))),
            'scanner_findings': [dict(f, file=Path(f['file']).as_posix()) for f in scan(path)],
        }
    out = ROOT / 'reports/pilot/vanna_data_case.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
