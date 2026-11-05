"""
Breast Cancer Diagnosis: end-to-end supervised ML pipeline
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), bundled with scikit-learn.

Pipeline:
  1. EDA (class balance, feature stats, correlation)
  2. Stratified train/test split (test set untouched until the very end)
  3. Compare 6 models with 5-fold stratified CV on the training set
  4. Hyperparameter tuning (GridSearchCV) for the top models
  5. Pick a decision threshold on out-of-fold predictions (no test leakage)
  6. Final one-shot evaluation on the held-out test set
  7. Interpretation: permutation importance

Convention: malignant = positive class (1). sklearn encodes malignant as 0,
so the label is flipped.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# Colorblind-safe palette (Okabe-Ito subset), thin marks, light grid
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
     "grey": "#6b7280", "purple": "#CC79A7", "yellow": "#E69F00"}
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5,
})

results = {}

# ---------------------------------------------------------------- 1. data
data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)          # 1 = malignant
results["n_samples"], results["n_features"] = X.shape
results["n_malignant"] = int(y.sum())
results["n_benign"] = int((1 - y).sum())
results["missing_values"] = int(X.isna().sum().sum())

# EDA fig: class balance + top separating features
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
counts = [results["n_benign"], results["n_malignant"]]
b = ax[0].bar(["Benign", "Malignant"], counts, color=[C["blue"], C["orange"]], width=0.5)
for r, v in zip(b, counts):
    ax[0].text(r.get_x() + r.get_width() / 2, v + 5, f"{v} ({v / len(y):.1%})",
               ha="center", fontsize=9)
ax[0].set_title("Class balance")
ax[0].set_ylabel("Patients")
ax[0].set_ylim(0, max(counts) * 1.15)

corr_with_y = X.corrwith(y).abs().sort_values(ascending=False)
top = corr_with_y.head(8)[::-1]
ax[1].barh(top.index, top.values, color=C["blue"], height=0.6)
ax[1].set_title("Top 8 features by |correlation| with malignancy")
ax[1].set_xlabel("|Pearson r|")
plt.tight_layout()
plt.savefig(FIG / "01_eda.png")
plt.close()
results["top_correlated_features"] = {k: round(float(v), 3) for k, v in corr_with_y.head(5).items()}

# redundancy: highly correlated feature pairs
cm = X.corr().abs()
upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
results["feature_pairs_corr_gt_0.9"] = int((upper > 0.9).sum().sum())

# ---------------------------------------------------------------- 2. split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
results["n_train"], results["n_test"] = len(X_tr), len(X_te)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def pipe(model):
    return Pipeline([("scale", StandardScaler()), ("clf", model)])

# ---------------------------------------------------------------- 3. compare
models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=5000, random_state=SEED)),
    "KNN (k=5)": pipe(KNeighborsClassifier()),
    "SVM (RBF)": pipe(SVC(probability=True, random_state=SEED)),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
    "MLP (neural net)": pipe(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                           random_state=SEED)),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for name, m in models.items():
    s = cross_validate(m, X_tr, y_tr, cv=cv, scoring=scoring)
    rows.append({"Model": name, **{k: s[f"test_{k}"].mean() for k in scoring},
                 "f1_std": s["test_f1"].std()})
cmp_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
cmp_df.round(4).to_csv(OUT / "model_comparison_cv.csv", index=False)
results["cv_comparison"] = cmp_df.round(4).to_dict(orient="records")

fig, ax = plt.subplots(figsize=(8, 3.8))
order = cmp_df["Model"][::-1]
vals = cmp_df.set_index("Model").loc[order]
ax.barh(order, vals["f1"], xerr=vals["f1_std"], color=C["blue"], height=0.55,
        error_kw=dict(ecolor=C["grey"], lw=1, capsize=3))
for i, v in enumerate(vals["f1"]):
    ax.text(0.9005, i, f"{v:.3f}", va="center", fontsize=9, color="white", fontweight="bold")
ax.set_xlim(0.9, 1.0)
ax.set_xlabel("Mean F1 (5-fold CV on training set; bars = 1 std)")
ax.set_title("Model comparison (malignant = positive class)")
plt.tight_layout()
plt.savefig(FIG / "02_model_comparison.png")
plt.close()

# ---------------------------------------------------------------- 4. tune
grids = {
    "SVM (RBF)": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 2, 4]},
}
tuned = {}
results["tuning"] = {}
for name, g in grids.items():
    gs = GridSearchCV(models[name], g, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuned[name] = gs.best_estimator_
    results["tuning"][name] = {
        "best_params": {k.replace("clf__", ""): (v if not isinstance(v, np.generic) else v.item())
                        for k, v in gs.best_params_.items()},
        "cv_f1": round(float(gs.best_score_), 4)}

best_name = max(results["tuning"], key=lambda k: results["tuning"][k]["cv_f1"])
results["selected_model"] = best_name
best = tuned[best_name]

# ---------------------------------------------------------------- 5. threshold
# Out-of-fold probabilities on the TRAINING set only -> no test leakage.
oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_tr, oof)
# Clinical rule: a missed cancer is far worse than a false alarm.
# Choose the highest threshold that still gives >= 98% recall on OOF data.
ok = np.where(rec[:-1] >= 0.98)[0]
chosen = float(thr[ok[-1]]) if len(ok) else 0.5
results["threshold"] = {"default": 0.5, "chosen_for_recall>=0.98": round(chosen, 4)}

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(thr, prec[:-1], color=C["blue"], lw=2, label="Precision")
ax.plot(thr, rec[:-1], color=C["orange"], lw=2, label="Recall")
ax.axvline(0.5, color=C["grey"], ls=":", lw=1.2)
ax.axvline(chosen, color=C["green"], ls="--", lw=1.5)
ax.text(0.5, 0.52, " default 0.5", color=C["grey"], fontsize=8.5)
ax.text(chosen, 0.52, f" chosen {chosen:.2f}", color=C["green"], fontsize=8.5)
ax.set_xlabel("Decision threshold (probability of malignant)")
ax.set_ylabel("Score (out-of-fold, training set)")
ax.set_title(f"Precision-recall trade-off, {best_name}")
ax.legend(frameon=False, loc="lower left")
ax.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.savefig(FIG / "03_threshold.png")
plt.close()

# ---------------------------------------------------------------- 6. test
best.fit(X_tr, y_tr)
p_te = best.predict_proba(X_te)[:, 1]

def metrics(pred):
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {"accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

pred_default = (p_te >= 0.5).astype(int)
pred_chosen = (p_te >= chosen).astype(int)
results["test_default_threshold"] = metrics(pred_default)
results["test_chosen_threshold"] = metrics(pred_chosen)
results["test_roc_auc"] = round(float(roc_auc_score(y_te, p_te)), 4)
results["test_report_default"] = classification_report(
    y_te, pred_default, target_names=["benign", "malignant"], output_dict=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for a, pred, title in [(ax[0], pred_default, "Threshold 0.5"),
                       (ax[1], pred_chosen, f"Threshold {chosen:.2f} (recall-first)")]:
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                           display_labels=["Benign", "Malignant"]).plot(
        ax=a, cmap="Blues", colorbar=False, values_format="d")
    a.set_title(title)
    a.grid(False)
fpr, tpr, _ = roc_curve(y_te, p_te)
ax[2].plot(fpr, tpr, color=C["blue"], lw=2, label=f"AUC = {results['test_roc_auc']:.3f}")
ax[2].plot([0, 1], [0, 1], color=C["grey"], ls=":", lw=1)
ax[2].set_xlabel("False positive rate")
ax[2].set_ylabel("True positive rate")
ax[2].set_title("ROC curve (test set)")
ax[2].legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "04_test_results.png")
plt.close()

# ---------------------------------------------------------------- 7. interpret
pi = permutation_importance(best, X_te, y_te, n_repeats=30, random_state=SEED,
                            scoring="f1", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
imp_sd = pd.Series(pi.importances_std, index=X.columns)[imp.index]
results["top_permutation_importance"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}

top8 = imp.head(8)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.9))
ax.barh(top8.index, top8.values, xerr=imp_sd[top8.index].values, color=C["blue"],
        height=0.55, error_kw=dict(ecolor=C["grey"], lw=1, capsize=3))
ax.set_xlabel("Drop in test F1 when feature is shuffled (30 repeats)")
ax.set_title("Which measurements drive the prediction?")
plt.tight_layout()
plt.savefig(FIG / "05_feature_importance.png")
plt.close()

# ---------------------------------------------------------------- save
def default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=default))
print(json.dumps(results, indent=2, default=default))

# ---------------------------------------------------------------- website data
import shutil
SITE = OUT / "docs"
(SITE / "figures").mkdir(parents=True, exist_ok=True)
for f in FIG.glob("*.png"):
    shutil.copy(f, SITE / "figures" / f.name)
site_payload = {
    "results": results,
    "chosen_threshold": round(chosen, 4),
    "test": {"prob": [round(float(p), 4) for p in p_te],
             "label": [int(v) for v in y_te]},
}
(SITE / "data.js").write_text("window.DATA = " + json.dumps(site_payload, default=default) + ";")
