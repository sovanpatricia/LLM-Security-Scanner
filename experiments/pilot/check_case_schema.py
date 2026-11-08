"""Validate every pilot case.json. Standard library only; run from repo root."""
import copy
import json
import unittest
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'data/pilot/cases'
ORIGINS = {'original', 'mutated', 'repaired'}
ROLES = {'target', 'context'}
APP_KINDS = {'intentionally_vulnerable_lab', 'real_world_app'}
METHODS = {'auto_same_file', 'manual_cross_file'}
REQUIRED = {
    'schema_version', 'case_id', 'group_id', 'source', 'label', 'variants',
    'excerpts', 'context', 'data_flow', 'validation', 'review', 'split',
    'excluded_from_training', 'assumptions', 'limitations',
}


def validate_case(data, case_dir, repo_root=ROOT):
    """Return descriptive errors without importing or executing any case code."""
    errors = []

    def require(obj, fields, location):
        if not isinstance(obj, dict):
            errors.append(f'{location}: expected object')
            return {}
        for field in sorted(fields - obj.keys()):
            errors.append(f'{location}.{field}: missing required field')
        return obj

    def choice(value, allowed, location):
        if not isinstance(value, str) or value not in allowed:
            errors.append(f'{location}: expected one of {sorted(allowed)}')

    def strings(value, location):
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errors.append(f'{location}: expected list of strings')

    def file_exists(value, base, location):
        if not isinstance(value, str) or not value:
            errors.append(f'{location}: expected nonempty relative file path')
            return
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            errors.append(f'{location}: path must be relative')
            return
        target = (base / value).resolve()
        if not target.is_relative_to(base.resolve()):
            errors.append(f'{location}: path escapes its base directory')
        elif not target.is_file():
            errors.append(f'{location}: file does not exist: {value}')

    data = require(data, REQUIRED, 'case')
    if type(data.get('schema_version')) is not int or data.get('schema_version') != 1:
        errors.append('schema_version: expected integer 1')
    for field in ('case_id', 'group_id', 'split'):
        if not isinstance(data.get(field), str) or not data.get(field):
            errors.append(f'{field}: expected nonempty string')
    source = require(data.get('source'), {'url', 'commit', 'license', 'license_file', 'app_kind'}, 'source')
    choice(source.get('app_kind'), APP_KINDS, 'source.app_kind')
    file_exists(source.get('license_file'), case_dir, 'source.license_file')
    require(data.get('label'), {'owasp', 'cwe'}, 'label')
    variants = require(data.get('variants'), {'vulnerable', 'control'}, 'variants')
    for variant in ('vulnerable', 'control'):
        entry = require(variants.get(variant), {'file', 'origin', 'description'}, f'variants.{variant}')
        choice(entry.get('origin'), ORIGINS, f'variants.{variant}.origin')
        file_exists(entry.get('file'), case_dir, f'variants.{variant}.file')
    excerpts = data.get('excerpts')
    if not isinstance(excerpts, list):
        errors.append('excerpts: expected list')
        excerpts = []
    targets = 0
    for i, entry in enumerate(excerpts):
        loc = f'excerpts[{i}]'
        entry = require(entry, {'role', 'file', 'start_line', 'end_line', 'symbol'}, loc)
        choice(entry.get('role'), ROLES, loc + '.role')
        targets += entry.get('role') == 'target'
        start, end = entry.get('start_line'), entry.get('end_line')
        if type(start) is not int or type(end) is not int or start < 1 or end < 1:
            errors.append(f'{loc}: line numbers must be positive integers')
        elif start > end:
            errors.append(f'{loc}: start_line > end_line')
    if targets != 1:
        errors.append('excerpts: expected exactly one target')
    context = require(data.get('context'), {'method', 'artifact'}, 'context')
    choice(context.get('method'), METHODS, 'context.method')
    if context.get('method') == 'auto_same_file':
        require(context, {'extractor', 'root_symbol', 'artifact'}, 'context')
        if context.get('extractor') != 'same-file-name-closure-v1':
            errors.append('context.extractor: expected same-file-name-closure-v1')
    require(data.get('data_flow'), {'source', 'sink'}, 'data_flow')
    validation = require(data.get('validation'), {'script', 'report', 'live_llm', 'summary'}, 'validation')
    for field in ('script', 'report'):
        file_exists(validation.get(field), repo_root, f'validation.{field}')
    if type(validation.get('live_llm')) is not bool:
        errors.append('validation.live_llm: expected boolean')