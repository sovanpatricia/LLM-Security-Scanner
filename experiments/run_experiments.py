"""Lab 3 experiments: EDA, template-disjoint split, tuning on validation, test, CV.

Run from the repository root:
    python experiments/run_experiments.py
Outputs: reports/results.json, reports/eda.json, reports/figures/*.png, models/best_model.joblib
"""
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llmscan.features import code_tokens, strip_comments  # noqa: E402
from llmscan.models import GRIDS, PRETTY, build  # noqa: E402
from llmscan.rules import rule_predict  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

LABELS = ["LLM01", "LLM02", "LLM03", "LLM05", "LLM06", "LLM07", "LLM08", "LLM10", "SAFE"]


# ---------------------------------------------------------------- data + split
def template_split(df, val_frac=0.25, seed=SEED):
    """Assign whole templates of the development set to train or validation, per class.

    No template is shared between train and validation, so validation measures
    generalisation to new code shapes, not memorisation of a template.
    """
    rng = random.Random(seed)
    split = {}
    for label, g in df.groupby("label"):
        temps = sorted(g.template.unique())
        rng.shuffle(temps)
        n_val = max(1, round(len(temps) * val_frac))
        for t in temps[:n_val]:
            split[t] = "val"
        for t in temps[n_val:]:
            split[t] = "train"
    return df.template.map(split)


def metrics(y_true, y_pred):
    yt_bin = [int(y != "SAFE") for y in y_true]
    yp_bin = [int(y != "SAFE") for y in y_pred]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "detect_precision": precision_score(yt_bin, yp_bin, zero_division=0),
        "detect_recall": recall_score(yt_bin, yp_bin, zero_division=0),
        "detect_f1": f1_score(yt_bin, yp_bin, zero_division=0),
    }


# ---------------------------------------------------------------- EDA
def eda(df):
    out = {"n": len(df), "n_templates": int(df.template.nunique())}
    out["per_class"] = df.label.value_counts().reindex(LABELS).to_dict()
    out["per_language"] = df.language.value_counts().to_dict()
    out["templates_per_class"] = df.groupby("label").template.nunique().reindex(LABELS).to_dict()
    df = df.assign(lines=df.code.str.count("\n") + 1,
                   tokens=df.code.map(lambda c: len(code_tokens(strip_comments(c)))))
    out["lines"] = df.lines.describe().round(1).to_dict()
    out["tokens_by_class"] = df.groupby("label").tokens.median().reindex(LABELS).to_dict()
    out["lang_by_class"] = pd.crosstab(df.label, df.language).reindex(LABELS).to_dict(orient="index")

    # most class-specific tokens (chi-squared, one-vs-rest)
    cv = CountVectorizer(tokenizer=code_tokens, token_pattern=None, lowercase=False, binary=True)
    X = cv.fit_transform(df.code.map(strip_comments))
    vocab = np.array(cv.get_feature_names_out())
    top = {}
    for lab in LABELS:
        scores, _ = chi2(X, (df.label == lab).astype(int))
        scores = np.nan_to_num(scores)
        present = np.asarray(X[(df.label == lab).values].mean(axis=0)).ravel()
        cand = [(s, t) for s, t, p in zip(scores, vocab, present) if p > 0.3 and len(t) > 2]
        top[lab] = [t for _, t in sorted(cand, reverse=True)[:6]]
    out["top_tokens"] = top
    out["vocab_size"] = len(vocab)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    df.label.value_counts().reindex(LABELS).plot.bar(ax=ax[0], color="#5b7fbf")
    ax[0].set_title("Snippets per class"); ax[0].set_ylabel("snippets")
    df.boxplot(column="tokens", by="label", ax=ax[1], grid=False)
    ax[1].set_title("Code tokens per snippet"); ax[1].set_xlabel(""); fig.suptitle("")
    fig.tight_layout(); fig.savefig(FIG / "eda.png", dpi=150); plt.close(fig)
    return out


# ---------------------------------------------------------------- experiments
def main():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    eda_res = eda(df)
    (REPORTS / "eda.json").write_text(json.dumps(eda_res, indent=2))

    te = pd.read_csv(ROOT / "data" / "testset.csv")  # held-out, written after rules/models were frozen
    df["split"] = template_split(df)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    results = {"split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
               "split_templates": {"train": int(tr.template.nunique()), "val": int(va.template.nunique()),
                                   "test": int(te.template.nunique())},
               "test_per_class": te.label.value_counts().reindex(LABELS).to_dict(),
               "models": {}}

    # 1) rules baseline (no training)
    clean = lambda s: [strip_comments(c) for c in s]  # noqa: E731
    results["models"]["rules"] = {"best_hp": None,
                                  "val": metrics(va.label, [rule_predict(c) for c in clean(va.code)]),
                                  "test": metrics(te.label, [rule_predict(c) for c in clean(te.code)])}
    preds = {"rules": [rule_predict(c) for c in clean(te.code)]}

    # 2) ML models: grid search on validation, refit on train+val, score on test
    trval = pd.concat([tr, va])
    for name, grid in GRIDS.items():
        scored = []
        for hp in grid:
            m = build(name, **hp).fit(tr.code, tr.label)
            scored.append((metrics(va.label, m.predict(va.code))["macro_f1"], hp))
        best_f1, best_hp = max(scored, key=lambda s: s[0])
        val_m = metrics(va.label, build(name, **best_hp).fit(tr.code, tr.label).predict(va.code))
        final = build(name, **best_hp).fit(trval.code, trval.label)
        p = final.predict(te.code)
        preds[name] = list(p)
        results["models"][name] = {"best_hp": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_hp.items()},
                                   "val": val_m, "test": metrics(te.label, p),
                                   "grid_size": len(grid)}
        print(f"{name:15s} val macroF1={val_m['macro_f1']:.3f} test macroF1={results['models'][name]['test']['macro_f1']:.3f} hp={best_hp}")
    print(f"rules           test macroF1={results['models']['rules']['test']['macro_f1']:.3f}")

    # 3) robustness: 5-fold stratified *group* CV over the whole dataset, best hp per model
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = {k: [] for k in ["rules", *GRIDS]}
    for tr_idx, te_idx in sgkf.split(df.code, df.label, groups=df.template):
        a, b = df.iloc[tr_idx], df.iloc[te_idx]
        cv["rules"].append(metrics(b.label, [rule_predict(c) for c in clean(b.code)])["macro_f1"])
        for name in GRIDS:
            hp = results["models"][name]["best_hp"]
            hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}
            m = build(name, **hp).fit(a.code, a.label)
            cv[name].append(metrics(b.label, m.predict(b.code))["macro_f1"])
    for k, v in cv.items():
        results["models"][k]["cv_macro_f1_mean"] = float(np.mean(v))
        results["models"][k]["cv_macro_f1_std"] = float(np.std(v))

    # 4) leakage check: a naive random split, where the same template can sit in train and test
    rtr, rte = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    results["random_split"] = {}
    for name in GRIDS:
        hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in results["models"][name]["best_hp"].items()}
        m = build(name, **hp).fit(rtr.code, rtr.label)
        results["random_split"][name] = metrics(rte.label, m.predict(rte.code))
    results["random_split"]["rules"] = metrics(rte.label, [rule_predict(c) for c in clean(rte.code)])

    # 5) model chosen by the protocol (best validation macro-F1) vs the one that generalised best
    selected = max(GRIDS, key=lambda k: results["models"][k]["val"]["macro_f1"])
    best_test = max(GRIDS, key=lambda k: results["models"][k]["test"]["macro_f1"])
    results["selected_by_val"], results["best_on_test"] = selected, best_test
    results["per_class"] = {k: classification_report(te.label, preds[k], labels=LABELS, output_dict=True, zero_division=0)
                            for k in ["rules", *GRIDS]}
    for key in {selected, best_test}:
        cm = confusion_matrix(te.label, preds[key], labels=LABELS)
        results[f"confusion_{key}"] = cm.tolist()
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45); ax.set_yticks(range(len(LABELS)), LABELS)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Test set: {PRETTY[key]}", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / f"confusion_{key}.png", dpi=150); plt.close(fig)

    ex = te.assign(pred=preds[best_test])
    ok, bad = ex[ex.label == ex.pred], ex[ex.label != ex.pred]
    cols = ["id", "template", "label", "pred", "code"]
    results["examples_correct"] = ok[ok.label != "SAFE"].sample(min(4, len(ok)), random_state=1)[cols].to_dict(orient="records")
    results["examples_wrong"] = bad[cols].to_dict(orient="records")
    results["test_predictions"] = te[["id", "template", "label"]].assign(**{k: preds[k] for k in preds}).to_dict(orient="records")

    # the model used by the app: trained on the whole development set (the test set stays unseen)
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in results["models"][best_test]["best_hp"].items()}
    joblib.dump(build(best_test, **hp).fit(df.code, df.label), ROOT / "models" / "scanner_model.joblib")
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("selected by val:", selected, "| best on test:", best_test)


if __name__ == "__main__":
    main()
