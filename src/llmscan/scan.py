"""Command-line scanner: split source files into functions, classify each one.

Usage:
    python -m llmscan.scan <file-or-folder> [--format md|json] [--model models/scanner_model.joblib]

Each Python function (or TypeScript function / exported constant) is passed to the
trained model. Snippets predicted as a risk are reported with file, line,
OWASP category, a confidence score and any rule that also fired.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

from .rules import rule_hits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "scanner_model.joblib"

OWASP = {
    "LLM01": "Prompt injection", "LLM02": "Sensitive information disclosure",
    "LLM03": "Supply chain", "LLM05": "Improper output handling", "LLM06": "Excessive agency",
    "LLM07": "System prompt leakage", "LLM08": "Vector and embedding weaknesses",
    "LLM10": "Unbounded consumption",
}
FIX = {
    "LLM01": "Keep user text in its own user message, validate it, and never concatenate it into instructions.",
    "LLM02": "Send the model only the fields it needs; redact PII and never log or return secrets.",
    "LLM03": "Pin models to a revision/hash, avoid trust_remote_code and pickle, and pin dependencies.",
    "LLM05": "Treat model output as untrusted: no eval/exec/shell, parameterised SQL, escape HTML.",
    "LLM06": "Give agents read-only, narrowly scoped tools and require human approval for actions.",
    "LLM07": "Move credentials and internal rules out of the system prompt; load secrets from the environment in tools.",
    "LLM08": "Filter every vector query by tenant/user and restrict who can add documents.",
    "LLM10": "Set max tokens, timeouts, retry limits, input size limits and rate limits.",
}
LLM_HINT = re.compile(r"llm|openai|anthropic|chat|completion|generate|invoke|prompt|agent|embed|vector|"
                      r"mistral|gemini|cohere|ollama|from_pretrained|tool", re.I)


def python_units(path: Path, src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [(1, src)]
    lines = src.splitlines()
    units = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue  # docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.Expr)):
            if isinstance(node, (ast.Assign, ast.Expr)) and node.end_col_offset < 40 and node.lineno == node.end_lineno:
                continue  # one-line set-up such as client = OpenAI()
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            units.append((start, "\n".join(lines[start - 1:node.end_lineno])))
    return units


def ts_units(path: Path, src: str):
    """Rough split of TypeScript/JavaScript into top-level declarations by brace matching."""
    starts = [m.start() for m in re.finditer(r"^(export\s+|async\s+|const\s+|function\s+|let\s+)", src, re.M)]
    units = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(src)
        units.append((src.count("\n", 0, s) + 1, src[s:e].strip()))
    return units or [(1, src)]


def iter_units(target: Path):
    files = [target] if target.is_file() else [p for p in target.rglob("*")
                                                 if p.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}
                                                 and "node_modules" not in p.parts]
    for f in sorted(files):
        src = f.read_text(encoding="utf-8", errors="ignore")
        units = python_units(f, src) if f.suffix == ".py" else ts_units(f, src)
        for line, code in units:
            if LLM_HINT.search(code):  # only code that touches an LLM
                yield f, line, code


def score(model, code):
    """Probability for probabilistic models; for the SVM, the margin between the top two classes."""
    if hasattr(model, "predict_proba"):
        return float(np.max(model.predict_proba([code])))
    s = np.sort(model.decision_function([code])[0])
    return float(s[-1] - s[-2])


def scan(target: Path, model_path: Path = DEFAULT_MODEL):
    model = joblib.load(model_path)
    findings = []
    for f, line, code in iter_units(target):
        label = model.predict([code])[0]
        if label == "SAFE":
            continue
        findings.append({"file": str(f), "line": line, "category": label, "risk": OWASP[label],
                         "score": round(score(model, code), 2),
                         "rules_agree": [r.name for r in rule_hits(code) if r.category == label],
                         "fix": FIX[label], "snippet": code.splitlines()[0][:100]})
    return findings


def scorecard(findings):
    return {cat: ("FAIL" if any(f["category"] == cat for f in findings) else "PASS") for cat in OWASP}


def to_markdown(findings):
    out = ["# LLM security scan", "", "| File:line | OWASP | Score | Suggested fix |", "| --- | --- | --- | --- |"]
    for f in findings:
        out.append(f"| `{Path(f['file']).name}:{f['line']}` | {f['category']} {f['risk']} | {f['score']} | {f['fix']} |")
    out += ["", "## Scorecard", "", "| Category | Status |", "| --- | --- |"]
    out += [f"| {c} {OWASP[c]} | {s} |" for c, s in scorecard(findings).items()]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args(argv)
    findings = scan(Path(a.target), Path(a.model))
    if a.format == "json":
        print(json.dumps({"findings": findings, "scorecard": scorecard(findings)}, indent=2))
    else:
        print(to_markdown(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
