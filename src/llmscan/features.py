"""Turning source code into features the models can learn from."""
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .rules import RULES

_PY_COMMENT = re.compile(r"#[^\n]*")
_JS_COMMENT = re.compile(r"(?<![:\"'])//[^\n]*")
_STRING = re.compile(r"(\"\"\"[\s\S]*?\"\"\"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[\s\S]*?`)")


def strip_comments(code: str) -> str:
    """Remove comments so the model cannot learn from hints such as '# no filter'."""
    code = _PY_COMMENT.sub("", code)
    code = _JS_COMMENT.sub("", code)
    return code


# Code-aware tokens: identifiers, numbers, and the punctuation that matters for
# security ( f" , + , ${ , = , ( ... ). Identifiers are also split on '_' and '.'
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\$\{|f\"|f'|==|=>|\*\*|[+=(){}\[\],.:`]")


def code_tokens(code: str):
    toks = []
    for t in TOKEN_RE.findall(code):
        toks.append(t.lower())
        if "_" in t and len(t) > 1:
            toks.extend(p.lower() for p in t.split("_") if p)
    return toks


class RuleHits(BaseEstimator, TransformerMixin):
    """One binary feature per hand-written rule (used by the hybrid model)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [[1.0 if r.matches(code) else 0.0 for r in RULES] for code in X]
        return sparse.csr_matrix(np.array(rows))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"rule:{r.name}" for r in RULES])
