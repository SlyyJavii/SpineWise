import os
import numpy as np
import pandas as pd
import optuna
from scipy.special import softmax
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
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
y = y_str.map(LABEL_TO_ID)
if y.isna().any():
    bad = y_str[y.isna()].unique().tolist()
    raise ValueError(f"Unknown labels found: {bad}")
y = y.astype(int).to_numpy()

users = df["user_id"].astype(str).fillna("NA")
n_users = users.nunique()


# pick folds safely 
requested_folds = 5
sessions = df["session_id"].astype(str).fillna("NA")
n_sessions = sessions.nunique()

# never exceed #sessions
n_splits = min(requested_folds, max(2, n_users))  # at least 2, at most n_users
# if some users are tiny/class-imbalanced SGKfold may fail to stratify at high n_splits so we back off gradually if needed
while True:
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(cv.split(X, y, groups=users))
        print(f"[CV] Using StratifiedGroupKFold grouped by user_id with n_splits={n_splits} over {n_users} users")
        break
    except ValueError as e:
        n_splits -= 1
        if n_splits < 2:
            raise RuntimeError(f"Could not create valid user-grouped folds: {e}")

# guard against any empty folds (paranoia)
valid_splits = []
for tr, te in cv_splits:
    if len(tr) == 0 or len(te) == 0:
        print("[CV] Skipping an empty fold")
        continue
    valid_splits.append((tr, te))
if not valid_splits:
    raise RuntimeError("All folds were empty; check user_id and class balance.")
cv_splits = valid_splits

def apply_decision_layer(decision_params, logits, raw_score=False):
    eps = decision_params["eps"]
    alpha = decision_params["alpha"]
    beta = decision_params["beta"]
    tau = decision_params["tau"]
    bias_mod = decision_params["bias_mod"]

    new_logits = logits / tau
    new_logits[:, 1] += bias_mod
    proba = softmax(new_logits, axis=1)

    C = [
        [0.0, 1 + alpha, 1.0],
        [beta, 0.0, beta],
        [1.0, 1 + alpha, 0.0]
    ]

    expected = proba @ np.array(C).T

    if raw_score:
        return expected

    y_argmin = expected.argmin(1)

    close_to_top = (proba.max(1) - proba[:, 1] <= eps)
    return np.where(close_to_top, 1, y_argmin)

def evaluation_contract(lgbm_params, decision_params, callback=None):
    scores = []

    for tr, va in cv_splits:
        model = LGBMClassifier(**lgbm_params)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="multi_logloss", callbacks=[early_stopping(stopping_rounds=100)])

        logits = model.predict(X.iloc[va], raw_score=True)
        yhat_val = apply_decision_layer(decision_params, logits)

        if callback: # runs once per fold
            callback(y[va], logits, yhat_val, model.best_iteration_)

        macro_f1 = f1_score(y[va], yhat_val, average="macro")
        scores.append(macro_f1)
    return np.mean(scores)

def objective(trial):
    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 600, 3000, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 200),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "max_bin": trial.suggest_int("max_bin", 63, 255),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    decision_params = {
        "eps" : trial.suggest_float("eps", 0.0, 0.02),
        "alpha" : trial.suggest_float("alpha", 0.3, 1.2),
        "beta" : trial.suggest_float("beta", 0.7, 1.5),
        "tau" : trial.suggest_float("tau", 0.9, 1),  # temperature
        "bias_mod" : trial.suggest_float("bias_mod", 0, 0.03)  # class-1 logit bias
    }
    return evaluation_contract(lgbm_params, decision_params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

best_lgbm_params = study.best_params
best_decision_params = {key: best_lgbm_params.pop(key) for key in ("eps", "alpha", "beta", "tau", "bias_mod")}

out_of_fold = {
    "logits": [],
    "y_vals": [],
}
best_iterations = []

# honest CV evaluation with the best params
def summary(y_va, logits, yhat_val, best_iteration):
    out_of_fold["logits"].extend(logits)
    out_of_fold["y_vals"].extend(y_va)
    best_iterations.append(best_iteration)

new_macro_f1 = evaluation_contract(best_lgbm_params, best_decision_params, callback=summary)

calibrator = LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    C=1.0,
    max_iter=1000,
    random_state=42
)

calibrator.fit(out_of_fold["logits"],
               out_of_fold["y_vals"])

best_lgbm_params["n_estimators"] = int(np.round(np.mean(best_iterations)))
best_clf = LGBMClassifier(**best_lgbm_params)
best_clf.fit(X, y)

os.makedirs("models", exist_ok=True)
bundle_path = "models/posture_lgbm_classifier.pkl"
joblib.dump(
    {"model": best_clf, "feature_names": FEATURES, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL, "calibrator": calibrator, "decision_params": best_decision_params},
    bundle_path
)

print(best_lgbm_params, best_decision_params)

# convert lists to arrays
y_true = np.array(out_of_fold["y_vals"])
logits = np.array(out_of_fold["logits"])
y_pred_raw = np.argmax(logits, axis=1)

y_pred_cal = calibrator.predict(logits)

# --- uncalibrated performance
raw_report = classification_report(
    y_true, y_pred_raw,
    labels=[0, 1, 2],
    target_names=["bad", "moderate", "good"],
    digits=3,
    zero_division=0
)
raw_f1 = f1_score(y_true, y_pred_raw, average="macro")

# --- calibrated performance
cal_report = classification_report(
    y_true, y_pred_cal,
    labels=[0, 1, 2],
    target_names=["bad", "moderate", "good"],
    digits=3,
    zero_division=0
)
cal_f1 = f1_score(y_true, y_pred_cal, average="macro")

# --- confusion matrix (calibrated)
cm = confusion_matrix(y_true, y_pred_cal, labels=[0, 1, 2])

print("\n================ FINAL SUMMARY ================")
print(f"Global macro-F1 (uncalibrated): {raw_f1:.3f}")
print(f"Global macro-F1 (calibrated):   {cal_f1:.3f}")
print("\n--- Calibrated per-class report ---")
print(cal_report)
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print(f"\nClass distribution (true labels): {np.bincount(y_true)}")
# ==========================================================