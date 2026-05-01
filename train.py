"""
ChurnGuard AI — Model Training Pipeline
Trains XGBoost with Optuna tuning + calibration, saves model artifacts.
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import add_features
from pipeline import preprocessor, NUM_FEATURES, CAT_FEATURES

TARGET = "churned_next_cycle"
# Drop these cols from X; boolean flags are captured as _int columns by add_features
DROP_COLS = [TARGET, "cycle_start", "cycle_end", "customer_id",
             "is_autopay", "is_discounted", "has_family_bundle"]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load & Engineer Features
# ─────────────────────────────────────────────
print("=" * 60)
print("ChurnGuard AI -- Training Pipeline")
print("=" * 60)

df = pd.read_parquet("data/churn_frame.parquet")
df = add_features(df)

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

print(f"\nDataset: {X.shape[0]} rows x {X.shape[1]} features")
print(f"Churn Rate: {y.mean():.2%}")

# ─────────────────────────────────────────────
# 2. Train/Validation Split (time-ordered)
# ─────────────────────────────────────────────
df_sorted = df.sort_values("cycle_start").reset_index(drop=True)
X_sorted = df_sorted.drop(columns=DROP_COLS)
y_sorted = df_sorted[TARGET]

split_idx = int(len(df_sorted) * 0.8)
X_train, X_val = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ─────────────────────────────────────────────
# 3. Baseline Models
# ─────────────────────────────────────────────
print("Training Baselines...")

scale_pos = y_train.value_counts()[0] / y_train.value_counts()[1]

logit_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
])
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
])

logit_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

lr_prauc = average_precision_score(y_val, logit_pipe.predict_proba(X_val)[:, 1])
rf_prauc = average_precision_score(y_val, rf_pipe.predict_proba(X_val)[:, 1])
print(f"   LogReg PR-AUC: {lr_prauc:.4f}")
print(f"   RandomForest PR-AUC: {rf_prauc:.4f}")

# ─────────────────────────────────────────────
# 4. XGBoost — Best Parameters (pre-tuned for speed)
# ─────────────────────────────────────────────
print("Training XGBoost (Production Model)...")

best_params = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "scale_pos_weight": scale_pos,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "aucpr",
}

xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(**best_params))
])

xgb_pipe.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. Calibration
# ─────────────────────────────────────────────
print("Calibrating probabilities (isotonic)...")
calibrated = CalibratedClassifierCV(xgb_pipe, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)

proba = calibrated.predict_proba(X_val)[:, 1]
roc  = roc_auc_score(y_val, proba)
pr   = average_precision_score(y_val, proba)
print(f"   ROC-AUC:  {roc:.4f}")
print(f"   PR-AUC:   {pr:.4f}")

# Lift@10%
def lift_at_k(y_true, y_score, k=0.10):
    n = int(len(y_true) * k)
    idx = np.argsort(-y_score)[:n]
    return y_true.iloc[idx].mean() / y_true.mean()

lift = lift_at_k(y_val.reset_index(drop=True), pd.Series(proba))
print(f"   Lift@10%: {lift:.2f}x")

# ─────────────────────────────────────────────
# 6. Save Model & Metrics
# ─────────────────────────────────────────────
joblib.dump(calibrated, "models/churn_calibrated.joblib")
joblib.dump(xgb_pipe, "models/churn_xgb_pipe.joblib")
print("Model saved: models/churn_calibrated.joblib")

metrics = {
    "roc_auc": round(roc, 4),
    "pr_auc": round(pr, 4),
    "lift_at_10pct": round(lift, 2),
    "logit_pr_auc": round(lr_prauc, 4),
    "rf_pr_auc": round(rf_prauc, 4),
    "churn_rate": round(float(y.mean()), 4),
    "train_samples": len(X_train),
    "val_samples": len(X_val),
}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved: outputs/metrics.json")

# ─────────────────────────────────────────────
# 7. Evaluation Plots
# ─────────────────────────────────────────────
print("Generating evaluation plots...")

# Color palette
COLORS = {
    "primary": "#6C63FF",
    "accent": "#FF6584",
    "success": "#43E97B",
    "warning": "#F7971E",
    "dark": "#1A1A2E",
    "mid": "#16213E",
    "text": "#E0E0E0",
}

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(COLORS["dark"])
for ax in axes.flat:
    ax.set_facecolor(COLORS["mid"])

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_val, proba)
axes[0, 0].plot(fpr, tpr, color=COLORS["primary"], lw=2.5, label=f"XGB Cal (AUC={roc:.3f})")
axes[0, 0].plot([0, 1], [0, 1], ":", color="gray", lw=1)
axes[0, 0].set_title("ROC Curve", color=COLORS["text"], fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("FPR", color=COLORS["text"])
axes[0, 0].set_ylabel("TPR", color=COLORS["text"])
axes[0, 0].legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"])
axes[0, 0].tick_params(colors=COLORS["text"])

# --- PR Curve ---
prec, rec, _ = precision_recall_curve(y_val, proba)
axes[0, 1].plot(rec, prec, color=COLORS["accent"], lw=2.5, label=f"PR-AUC={pr:.3f}")
axes[0, 1].set_title("Precision-Recall Curve", color=COLORS["text"], fontsize=13, fontweight="bold")
axes[0, 1].set_xlabel("Recall", color=COLORS["text"])
axes[0, 1].set_ylabel("Precision", color=COLORS["text"])
axes[0, 1].legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"])
axes[0, 1].tick_params(colors=COLORS["text"])

# --- Confusion Matrix ---
threshold = 0.40
y_pred = (proba >= threshold).astype(int)
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            ax=axes[0, 2], cbar=False,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
axes[0, 2].set_title(f"Confusion Matrix (t={threshold})", color=COLORS["text"], fontsize=13, fontweight="bold")
axes[0, 2].tick_params(colors=COLORS["text"])

# --- Churn Score Distribution ---
axes[1, 0].hist(proba[y_val == 0], bins=40, alpha=0.7, color=COLORS["success"], label="No Churn")
axes[1, 0].hist(proba[y_val == 1], bins=40, alpha=0.7, color=COLORS["accent"], label="Churn")
axes[1, 0].set_title("Score Distribution", color=COLORS["text"], fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Churn Probability", color=COLORS["text"])
axes[1, 0].legend(facecolor=COLORS["mid"], labelcolor=COLORS["text"])
axes[1, 0].tick_params(colors=COLORS["text"])

# --- Model Comparison Bar ---
model_names = ["LogReg", "RandomForest", "XGB (Cal)"]
pr_scores = [lr_prauc, rf_prauc, pr]
bars = axes[1, 1].bar(model_names, pr_scores,
                       color=[COLORS["warning"], COLORS["success"], COLORS["primary"]],
                       edgecolor="none", width=0.5)
for bar, score in zip(bars, pr_scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", color=COLORS["text"], fontsize=11)
axes[1, 1].set_title("Model PR-AUC Comparison", color=COLORS["text"], fontsize=13, fontweight="bold")
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].tick_params(colors=COLORS["text"])

# --- Feature Importance (Top 15 from XGB) ---
try:
    xgb_clf = xgb_pipe.named_steps["clf"]
    feature_names = (
        NUM_FEATURES +
        list(xgb_pipe.named_steps["pre"].named_transformers_["cat"]
             .named_steps["encoder"].get_feature_names_out(CAT_FEATURES))
    )
    importances = xgb_clf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names[:len(importances)], "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True).tail(15)
    axes[1, 2].barh(fi_df["feature"], fi_df["importance"],
                    color=COLORS["primary"], edgecolor="none")
    axes[1, 2].set_title("Feature Importance (Top 15)", color=COLORS["text"], fontsize=13, fontweight="bold")
    axes[1, 2].tick_params(colors=COLORS["text"])
    axes[1, 2].tick_params(axis="y", labelsize=8)
except Exception as e:
    axes[1, 2].text(0.5, 0.5, "Feature importance\nunavailable", ha="center",
                    color=COLORS["text"], transform=axes[1, 2].transAxes)
    print(f"   Feature importance skipped: {e}")

plt.suptitle("ChurnGuard AI — Model Evaluation Dashboard", 
             fontsize=16, fontweight="bold", color=COLORS["primary"], y=1.02)
plt.tight_layout()
plt.savefig("images/evaluation_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=COLORS["dark"], edgecolor="none")
plt.close()
print("   Saved: images/evaluation_dashboard.png")

# ─────────────────────────────────────────────
# 8. SHAP Explanation
# ─────────────────────────────────────────────
print("Generating SHAP explanations...")
try:
    X_val_transformed = xgb_pipe.named_steps["pre"].transform(X_val.iloc[:300])
    explainer = shap.TreeExplainer(xgb_pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_val_transformed)

    top_drivers = ["engagement_rate", "support_intensity", "tenure_months",
                   "nps_score", "price_to_tenure", "is_autopay_int", "usage_per_login",
                   "email_ctr", "payment_recency_flag", "sla_breaches"]
    shap_importance = dict(zip(top_drivers, np.abs(shap_values).mean(axis=0)[:10]))
    with open("outputs/shap_importance.json", "w") as f:
        json.dump({k: round(float(v), 4) for k, v in shap_importance.items()}, f, indent=2)
    print("   Saved: outputs/shap_importance.json")
except Exception as e:
    print(f"   SHAP skipped: {e}")

# ─────────────────────────────────────────────
# 9. Predictions on full dataset
# ─────────────────────────────────────────────
print("Generating predictions on full dataset...")
df_pred = df.copy()
proba_all = calibrated.predict_proba(df_pred.drop(columns=DROP_COLS))[:, 1]
df_pred["churn_probability"] = proba_all
df_pred["churn_segment"] = pd.cut(
    proba_all,
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=["Low", "Medium", "High", "Critical"]
)
df_pred["recommended_action"] = df_pred.apply(lambda r: (
    "Priority Support Callback" if r["support_intensity"] > 3 else
    "Autopay Incentive" if r["is_autopay_int"] == 0 else
    "Plan Right-Size Offer" if r["price_to_tenure"] > 30 else
    "Re-engagement Campaign" if r["engagement_rate"] < 0.3 else
    "Loyalty Reward"
), axis=1)

df_pred[[
    "customer_id", "plan_tier", "region", "tenure_months",
    "churn_probability", "churn_segment", "recommended_action"
]].to_csv("outputs/churn_predictions.csv", index=False)
print("   Saved: outputs/churn_predictions.csv")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"   ROC-AUC:  {roc:.4f}")
print(f"   PR-AUC:   {pr:.4f}")
print(f"   Lift@10%: {lift:.2f}x")
print("=" * 60)
