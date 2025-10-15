import os
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import warnings

# config
#if youur csv is a different name then put the correct one 
CSV = "posture_dataset.csv"

METRICS = ["head_tilt","clavicle_drop_pct","face_lean","shoulder_ear_pct","torso_lean_pct","looking_down_pct"]
STATS   = ["mean","std","min","max","range","slope","delta"]

def build_feature_list(metrics=METRICS, stats=STATS):
    return [f"{m}_{s}" for m in metrics for s in stats]

FEATURES = build_feature_list()

NON_FEATURES = ["user_id","session_id","t_start","t_end","label"]

LABEL_TO_ID = {"bad": 0, "moderate": 1, "good": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# load + validate
df = pd.read_csv(CSV)

# create any missing non-features columns for safety
for c in NON_FEATURES:
    if c not in df.columns:
        df[c] = np.nan

# ensure features exist
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected feature columns: {missing}")

X = df[FEATURES].copy()
y_str = df["label"].astype(str)
y = y_str.map(LABEL_TO_ID).astype(int).to_numpy()

users = df["user_id"].astype(str).fillna("NA")
n_users = users.nunique()


# pick folds safely (session grouped) eventually user grouped
requested_folds = 5
sessions = df["session_id"].astype(str).fillna("NA")
n_sessions = sessions.nunique()

# never exceed #sessions
n_splits = min(requested_folds, max(2, n_sessions))  # at least 2, at most n_sessions
# if some sessions are tiny/class-imbalanced SGKfold may fail to stratify at high n_splits so we back off gradually if needed
while True:
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y, groups=sessions))
        print(f"[CV] Using StratifiedGroupKFold grouped by session_id with n_splits={n_splits} over {n_sessions} sessions")
        break
    except ValueError as e:
        n_splits -= 1
        if n_splits < 2:
            raise RuntimeError(f"Could not create valid session-grouped folds: {e}")

# guard against any empty folds (paranoia)
valid_splits = []
for tr, te in cv_splits:
    if len(tr) == 0 or len(te) == 0:
        print("[CV] Skipping an empty fold")
        continue
    valid_splits.append((tr, te))
if not valid_splits:
    raise RuntimeError("All folds were empty; check session_id and class balance.")
cv_splits = valid_splits


# guard against any empty folds 
valid_splits = []
for tr, te in cv_splits:
    if len(tr) == 0 or len(te) == 0:
        print("[CV] Skipping an empty fold")
        continue
    valid_splits.append((tr, te))
if not valid_splits:
    raise RuntimeError("All folds were empty; check CV setup.")
cv_splits = valid_splits


# model + search space
base_clf = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    boosting_type="gbdt",
    n_jobs=-1,
    random_state=42,
    learning_rate=0.07,
    n_estimators=600,
    max_depth=4,
    num_leaves=31,
    subsample=0.9,          # bagging_fraction
    subsample_freq=1,       # bagging_freq
    colsample_bytree=0.9,   # feature_fraction
    reg_alpha=0.0,          
    reg_lambda=0.0,         
    verbose=-1
)

cost = [
    [0, 2, 1],
    [1, 0, 1],
    [1, 2, 0]
]

def objective(trial): # reduce n_trials below if taking too long
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 600, 2000, step=10),
        "max_depth": trial.suggest_int("max_depth", -1, 8),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.3),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 2)
    }

    scores = []
    for tr, va in cv_splits:
        model = LGBMClassifier(**param, verbose=-1, objective="multiclass", num_class=3)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(stopping_rounds=100)])
        probs = model.predict_proba(X.iloc[va])
        yhat_val = np.argmin(probs @ cost, axis=1)
        macro_f1 = f1_score(y[va], yhat_val, average="macro")

        scores.append(macro_f1)

    return np.mean(scores)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
params = study.best_params
best_clf = LGBMClassifier(**params, objective="multiclass", num_class=3)

# honest CV evaluation with the best params
fold_f1 = []
per_class_f1 = []

for i, (train_idx, test_idx) in enumerate(cv_splits, 1):
    clf = LGBMClassifier(**params, objective="multiclass", num_class=3)
    clf.fit(X.iloc[train_idx], y[train_idx])
    y_pred = clf.predict(X.iloc[test_idx])

    f1_macro = f1_score(y[test_idx], y_pred, average="macro")
    fold_f1.append(f1_macro)

    report = classification_report(
        y[test_idx], y_pred,
        labels=[0,1,2],
        target_names=["bad","moderate","good"],
        output_dict=True,
        zero_division=0
    )
    per_class_f1.append([report["bad"]["f1-score"], report["moderate"]["f1-score"], report["good"]["f1-score"]])

    cm = confusion_matrix(y[test_idx], y_pred, labels=[0,1,2])
    print(f"\n[Fold {i}] macro-F1: {f1_macro:.3f}")
    print(f"[Fold {i}] confusion matrix (rows=true, cols=pred):")
    print(cm)

print("\n================ SUMMARY ================")
print(f"CV macro-F1 (classifier): {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
pcf = np.array(per_class_f1)
print(f"Per-class F1 avg: bad={pcf[:,0].mean():.3f}, moderate={pcf[:,1].mean():.3f}, good={pcf[:,2].mean():.3f}")

# fit best on full data and save with early stopping
best_clf.fit(
    X, y,
    eval_set = [(X,y)],
    eval_metric = "multi_logloss",
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

os.makedirs("models", exist_ok=True)
bundle_path = "models/posture_lgbm_classifier.pkl"
joblib.dump(
    {"model": best_clf, "feature_names": FEATURES, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL},
    bundle_path
)
print(f"\n[MODEL] Saved to {bundle_path}")
