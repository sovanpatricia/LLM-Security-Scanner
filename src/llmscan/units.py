"""Split Python source into function-level units (improvement 1, step 4).

The Lab 3 scanner kept a whole class as one unit, so in real applications,
where most code lives in classes, the model saw almost the whole file. Here a
unit is a top-level function, a method (dedented), or a top-level or class-level
assignment. Each unit keeps its line range so a finding can be located.
"""
import ast
import textwrap


def _span(node):
    start = min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])
    return start, node.end_lineno


def function_units(src):
    """Return a list of (start_line, end_line, kind, name, code)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        n = src.count('\n') + 1
        return [(1, n, 'module', '<module>', src)]
    lines = src.splitlines()
    units = []

    def add(node, kind, name):
        start, end = _span(node)
        units.append((start, end, kind, name, textwrap.dedent('\n'.join(lines[start - 1:end]))))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, 'function', node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, 'method', f'{node.name}.{item.name}')
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    add(item, 'class_assign', node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            add(node, 'assign', '<module>')
    return units
