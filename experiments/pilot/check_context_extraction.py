"""Small regression checks for context coverage, unrelated code and bounds."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from llmscan.context import extract_context


class ContextTests(unittest.TestCase):
    def test_dependency_chain_and_unrelated_function(self):
        source = ('import json\n'
                  'class Store:\n    def read(self):\n        return json.dumps([])\n'
                  'def wrapper():\n    return Store().read()\n'
                  'def unrelated():\n    return "DO_NOT_INCLUDE"\n'
                  'registered = wrapper\n')
        unit = extract_context(source, 'registered')
        self.assertIn('class Store:', unit.code)
        self.assertIn('def wrapper', unit.code)
        self.assertNotIn('DO_NOT_INCLUDE', unit.code)
        self.assertEqual(unit.spans[-1]['start_line'], 9)

    def test_recursive_dependency_terminates(self):
        unit = extract_context('def a():\n    return b()\ndef b():\n    return a()\n', 'a')
        self.assertEqual(len(unit.spans), 2)

    def test_missing_symbol_and_explicit_bounds(self):
        source = 'def a():\n    return b()\ndef b():\n    return 1\n'
        for kwargs in ({'root_symbol': 'missing'}, {'root_symbol': 'a', 'max_nodes': 1},
                       {'root_symbol': 'a', 'max_lines': 1}):
            with self.assertRaises(ValueError):
                extract_context(source, **kwargs)

    def test_pilot_pair_retains_security_relevant_difference(self):
        case = ROOT / 'data/pilot/cases/dvla_sensitive_tool_result'
        units = {v: extract_context((case / f'{v}.py').read_text(), 'get_current_user_tool')
                 for v in ('control', 'vulnerable')}
        self.assertIn('def get_current_user', units['control'].code)
        self.assertIn('def get_user(', units['control'].code)
        self.assertEqual(units['vulnerable'].code, units['control'].code.replace(
            'SELECT userId,username FROM Users', 'SELECT userId,username,password FROM Users'))


if __name__ == '__main__':
    unittest.main()
