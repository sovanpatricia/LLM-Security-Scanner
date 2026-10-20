"""Model definitions: each returns an sklearn Pipeline that takes raw code strings."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .features import RuleHits, code_tokens, strip_comments

SEED = 42


def _strip(X):
    return [strip_comments(c) for c in X]


def _token_tfidf(ngram=(1, 2), min_df=1):
    return TfidfVectorizer(tokenizer=code_tokens, token_pattern=None, lowercase=False,
                           ngram_range=ngram, min_df=min_df, sublinear_tf=True)


def build(name, **hp):
    clean = ("strip_comments", FunctionTransformer(_strip))
    if name == "tfidf_logreg":
        return Pipeline([clean, ("tfidf", _token_tfidf(hp.get("ngram", (1, 2)))),
                         ("clf", LogisticRegression(C=hp.get("C", 10), max_iter=5000,
                                                    class_weight="balanced", random_state=SEED))])
    if name == "char_svm":
        return Pipeline([clean, ("tfidf", TfidfVectorizer(analyzer="char_wb",
                                                          ngram_range=hp.get("ngram", (3, 5)),
                                                          sublinear_tf=True)),
                         ("clf", LinearSVC(C=hp.get("C", 1), class_weight="balanced", random_state=SEED))])
    if name == "random_forest":
        return Pipeline([clean, ("tfidf", _token_tfidf(hp.get("ngram", (1, 2)))),
                         ("clf", RandomForestClassifier(n_estimators=hp.get("n_estimators", 300),
                                                        max_depth=hp.get("max_depth"),
                                                        class_weight="balanced", random_state=SEED,
                                                        n_jobs=-1))])
    if name == "hybrid_logreg":
        feats = FeatureUnion([("tfidf", _token_tfidf(hp.get("ngram", (1, 2)))),
                              ("rules", RuleHits())],
                             transformer_weights={"tfidf": 1.0, "rules": hp.get("rule_weight", 1.0)})
        return Pipeline([clean, ("features", feats),
                         ("clf", LogisticRegression(C=hp.get("C", 10), max_iter=5000,
                                                    class_weight="balanced", random_state=SEED))])
    raise ValueError(name)


# Hyperparameter grids searched on the validation set
GRIDS = {
    "tfidf_logreg": [{"C": c, "ngram": n} for c in (0.1, 1, 10, 100) for n in ((1, 1), (1, 2), (1, 3))],
    "char_svm": [{"C": c, "ngram": n} for c in (0.1, 1, 10) for n in ((2, 4), (3, 5), (3, 6))],
    "random_forest": [{"n_estimators": e, "max_depth": d, "ngram": (1, 2)} for e in (100, 300, 500) for d in (None, 10, 30)],
    "hybrid_logreg": [{"C": c, "rule_weight": w, "ngram": (1, 2)} for c in (1, 10, 100) for w in (0.5, 1.0, 2.0, 4.0)],
}

PRETTY = {
    "rules": "Rules only (baseline)",
    "tfidf_logreg": "TF-IDF tokens + Logistic Regression",
    "char_svm": "TF-IDF char n-grams + Linear SVM",
    "random_forest": "TF-IDF tokens + Random Forest",
    "hybrid_logreg": "Hybrid: TF-IDF + rule hits + Logistic Regression",
}
