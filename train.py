import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CSV = "posture_dataset.csv"

# Metrics per frame/window
METRICS = ["head_tilt","clavicle_drop_pct","face_lean","shoulder_ear_pct","torso_lean_pct","looking_down_pct"]

# Stats per window
STATS = ["mean","std","min","max","range","slope","delta"]

def build_feature_list(metrics, stats):
    return [f"{m}_{s}" for m in metrics for s in stats]

df = pd.read_csv(CSV)
FEATURES = build_feature_list(METRICS, STATS)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected feature columns: {missing}")

# keep metadata for grouping/QA; exclude from X when fitting
NON_FEATURES = ["user_id","session_id","t_start","t_end","label"]
for c in NON_FEATURES:
    if c not in df.columns: df[c] = np.nan

ORDINAL_MAP = {"bad": 0, "moderate": 1, "good": 2}

X = df[FEATURES]
y = df["label"].map(ORDINAL_MAP).astype(int).to_numpy()
groups = df["user_id"].astype(str)

cv = GroupKFold(n_splits=groups.nunique())
splits = list(cv.split(X, y, groups=groups))

reg0 = LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    random_state=42,
    n_jobs=-1,
    learning_rate=0.07,
    n_estimators=800,
    num_leaves=31,
    max_depth=4,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbosity=-1,
)

cla0 = LGBMClassifier(objective="multiclass")

param_dist = {
    "learning_rate": np.geomspace(0.05, 0.1, 6),
    "n_estimators": np.linspace(600, 1000, 6, dtype=int),
    "max_depth": [4, 5],
    "num_leaves": [31],
    "min_child_samples": [20, 40],
    "min_child_weight": np.geomspace(1e-3, 1.0, 5),
    "min_split_gain": [0.0, 1e-3, 0.05, 0.1],
    "lambda_l1": np.geomspace(1e-4, 0.1, 6),
    "lambda_l2": np.geomspace(1e-3, 3.0, 6),
    "feature_fraction": [0.9, 1.0],
    "bagging_fraction": [0.8, 0.9],
    "bagging_freq": [1, 2],
}

search = RandomizedSearchCV(
    estimator=reg0,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_mean_squared_error",   # regression metric for search
    cv=splits,                          # <- group-aware
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=1,
)
search.fit(X, y)
best_reg = search.best_estimator_

search = RandomizedSearchCV(
    estimator=cla0,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_mean_squared_error",   # regression metric for search
    cv=splits,                          # <- group-aware
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=1,
)
search.fit(X, y)
best_reg_2 = search.best_estimator_

def apply_thresholds(y_hat, t1, t2):
    return np.where(y_hat <= t1, 0, np.where(y_hat >= t2, 1, 2))

def tune_thresholds(y_true, y_hat):
    t1_range = np.linspace(0.2, 1.2, 33)
    t2_range = np.linspace(1.0, 1.9, 37)
    best = (-1, 0.2, 1.0)
    for t1 in t1_range:
        for t2 in t2_range:
            if t2 <= t1:
                continue
            f1 = f1_score(y_true, apply_thresholds(y_hat, t1, t2), average="macro")
            if f1 > best[0]:
                best = (f1, t1, t2)
    return best

fold_f1, t1s, t2s = [], [], []
cla_f1 = []

for tr_idx, va_idx in splits:
    cla = LGBMClassifier(**best_reg_2.get_params())
    reg = LGBMRegressor(**best_reg.get_params())

    cla.fit(X.iloc[tr_idx], y[tr_idx])
    reg.fit(X.iloc[tr_idx], y[tr_idx])

    yhat_val_cla = cla.predict(X.iloc[va_idx])
    yhat_val = reg.predict(X.iloc[va_idx])

    f1, t1, t2 = tune_thresholds(y[va_idx], yhat_val)
    fold_f1.append(f1); t1s.append(t1); t2s.append(t2)

    cla_f1.append(f1_score(y[va_idx], yhat_val_cla, average="macro"))

cv_macro_f1 = float(np.mean(fold_f1))
t1_cv, t2_cv = float(np.median(t1s)), float(np.median(t2s))
print(f"Grouped-CV macro-F1 (REGRESSOR): {cv_macro_f1:.3f} with thresholds {t1_cv:.3f} and {t2_cv:.3f}")
print(f"Grouped-CV macro-F1 (CLASSIFIER): {float(np.mean(cla_f1)):.3f}")