"""Bounded, same-file Python context extraction for step 3 experiments.

Parse only: never imports or executes the application. This is a conservative
name-reference closure, not interprocedural data-flow or authorization analysis.
"""
import ast
from dataclasses import asdict, dataclass


@dataclass
class ContextUnit:
    root_symbol: str
    code: str
    spans: list[dict]
    external_imports: list[str]
    policy: str = "same-file-name-closure-v1"

    def to_dict(self):
        return asdict(self)


def extract_context(source: str, root_symbol: str, *, max_nodes=20, max_lines=300):
    """Include imports, a named declaration and its referenced local declarations.

    Whole classes are retained when referenced. Definitions are emitted once in
    original order, with source spans. Exceeding bounds raises rather than
    silently losing potentially relevant context. Imported helpers are NOT read.
    """
    tree = ast.parse(source)
    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [n.id for target in node.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            if name in definitions:
                raise ValueError(f"Ambiguous repeated top-level binding: {name}")
            definitions[name] = node
    if root_symbol not in definitions:
        raise ValueError(f"Unknown top-level symbol: {root_symbol}")

    pending = [definitions[root_symbol]]
    selected = set()
    while pending:
        node = pending.pop()
        if node in selected:
            continue
        selected.add(node)
        if len(selected) > max_nodes:
            raise ValueError("Context exceeds max_nodes; narrow the case or increase the explicit bound")
        for ref in ast.walk(node):
            if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load):
                dependency = definitions.get(ref.id)
                if dependency is not None and dependency not in selected:
                    pending.append(dependency)

    lines = source.splitlines()
    snippets, spans = [], []
    for node in sorted(selected | set(imports), key=lambda n: n.lineno):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        snippets.append("\n".join(lines[start - 1:node.end_lineno]))
        spans.append({"start_line": start, "end_line": node.end_lineno,
                      "kind": type(node).__name__})
    code = "\n\n".join(snippets) + "\n"
    if len(code.splitlines()) > max_lines:
        raise ValueError("Context exceeds max_lines; no code was silently truncated")
    external = [ast.get_source_segment(source, node) for node in imports]
    return ContextUnit(root_symbol, code, spans, external)
