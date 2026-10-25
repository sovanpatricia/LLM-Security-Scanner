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
# PROTOCOL (fixed before the final test set was evaluated, see git history):
#   1. Every model/hyperparameter setting is trained on the development set (199) and scored
#      by macro-F1 on the new-style validation set (63). That set was written after the rules
#      were frozen, so it is the only validation data the rules never saw.
#   2. Best setting per model = highest validation macro-F1 (ties: higher detection F1).
#      Chosen model = the model with the highest validation macro-F1, rules included.
#   3. The chosen model is retrained on development + new-style validation (262 snippets).
#   4. It is scored ONCE on the final test set (54). Test scores of the other models are
#      reported for transparency only; they never influence the choice.
def tuple_hp(hp):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}


def main():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    v2 = pd.read_csv(ROOT / "data" / "val_newstyle.csv")
    te = pd.read_csv(ROOT / "data" / "testset.csv")
    (REPORTS / "eda.json").write_text(json.dumps(eda(df), indent=2))
    clean = lambda s: [strip_comments(c) for c in s]  # noqa: E731
    rules_pred = lambda s: [rule_predict(c) for c in clean(s)]  # noqa: E731

    df["split"] = template_split(df)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    results = {"sizes": {"dev": len(df), "dev_train": len(tr), "dev_val": len(va),
                         "val_newstyle": len(v2), "test": len(te)},
               "templates": {"dev": int(df.template.nunique()), "val_newstyle": int(v2.template.nunique()),
                             "test": int(te.template.nunique())},
               "per_class": {"val_newstyle": v2.label.value_counts().reindex(LABELS).to_dict(),
                             "test": te.label.value_counts().reindex(LABELS).to_dict()},
               "models": {}}

    # step 1-2: tune and choose on the new-style validation set
    results["models"]["rules"] = {"best_hp": None, "val_newstyle": metrics(v2.label, rules_pred(v2.code)),
                                  "dev_val": metrics(va.label, rules_pred(va.code))}
    for name, grid in GRIDS.items():
        scored = []
        for hp in grid:
            m = build(name, **hp).fit(df.code, df.label)
            mt = metrics(v2.label, m.predict(v2.code))
            scored.append(((mt["macro_f1"], mt["detect_f1"]), hp, mt))
        _, best_hp, best_m = max(scored, key=lambda s: s[0])
        # for the report: the earlier (leaky) protocol, dev-train -> dev-val, same hyperparameters
        dv = metrics(va.label, build(name, **best_hp).fit(tr.code, tr.label).predict(va.code))
        results["models"][name] = {"best_hp": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_hp.items()},
                                   "grid_size": len(grid), "val_newstyle": best_m, "dev_val": dv}
        print(f"{name:15s} new-style val macroF1={best_m['macro_f1']:.3f}  (dev val {dv['macro_f1']:.3f})  hp={best_hp}")
    print(f"{'rules':15s} new-style val macroF1={results['models']['rules']['val_newstyle']['macro_f1']:.3f}")
    chosen = max(results["models"], key=lambda k: (results["models"][k]["val_newstyle"]["macro_f1"],
                                                   results["models"][k]["val_newstyle"]["detect_f1"]))
    results["chosen"] = chosen
    print("CHOSEN ON VALIDATION:", chosen)

    # step 3-4: retrain on dev + new-style validation, score once on the test set
    full = pd.concat([df.drop(columns="split"), v2])
    preds = {}
    for name in results["models"]:
        if name == "rules":
            preds[name] = rules_pred(te.code)
        else:
            m = build(name, **tuple_hp(results["models"][name]["best_hp"])).fit(full.code, full.label)
            preds[name] = list(m.predict(te.code))
            if name == chosen:
                joblib.dump(m, ROOT / "models" / "scanner_model.joblib")
        results["models"][name]["test"] = metrics(te.label, preds[name])
    print(f"TEST ({chosen}): macroF1={results['models'][chosen]['test']['macro_f1']:.3f} "
          f"detectF1={results['models'][chosen]['test']['detect_f1']:.3f}")

    # robustness: grouped 5-fold CV on the development set
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = {k: [] for k in results["models"]}
    for a_idx, b_idx in sgkf.split(df.code, df.label, groups=df.template):
        a, b = df.iloc[a_idx], df.iloc[b_idx]
        cv["rules"].append(metrics(b.label, rules_pred(b.code))["macro_f1"])
        for name in GRIDS:
            m = build(name, **tuple_hp(results["models"][name]["best_hp"])).fit(a.code, a.label)
            cv[name].append(metrics(b.label, m.predict(b.code))["macro_f1"])
    for k, v in cv.items():
        results["models"][k]["cv_mean"], results["models"][k]["cv_std"] = float(np.mean(v)), float(np.std(v))

    # leakage check: naive random split of the development set
    rtr, rte = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    results["random_split"] = {"rules": metrics(rte.label, rules_pred(rte.code))}
    for name in GRIDS:
        m = build(name, **tuple_hp(results["models"][name]["best_hp"])).fit(rtr.code, rtr.label)
        results["random_split"][name] = metrics(rte.label, m.predict(rte.code))

    # analysis of the chosen model on the test set
    results["per_class_test"] = {k: classification_report(te.label, preds[k], labels=LABELS, output_dict=True,
                                                          zero_division=0) for k in preds}
    cm = confusion_matrix(te.label, preds[chosen], labels=LABELS)
    results["confusion_test"] = cm.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45); ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Final test set: {PRETTY[chosen]}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "confusion_test.png", dpi=150); plt.close(fig)

    ex = te.assign(pred=preds[chosen])
    cols = ["id", "template", "label", "pred", "code"]
    results["examples_correct"] = ex[(ex.label == ex.pred) & (ex.label != "SAFE")][cols].to_dict(orient="records")
    results["examples_wrong"] = ex[ex.label != ex.pred][cols].to_dict(orient="records")
    results["test_predictions"] = te[["id", "template", "label"]].assign(**preds).to_dict(orient="records")
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
