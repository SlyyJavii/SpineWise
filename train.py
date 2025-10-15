import pandas as pd
import numpy as np
import datetime as dt
import optuna, joblib
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, mean_squared_error

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
groups = df["session_id"].astype(str)

cv = StratifiedGroupKFold(n_splits=groups.nunique())
splits = list(cv.split(X, y, groups=groups))

def objective(trial): # reduce n_trials below if taking too long
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
    }

    scores = []
    for tr, va in splits:
        model = LGBMClassifier(**param, verbose=-1)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[early_stopping(stopping_rounds=100)])
        pred = model.predict(X.iloc[va])
        rmse = mean_squared_error(y[va], pred)
        scores.append(rmse)

    return np.mean(scores)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

params = study.best_params
final_cla = LGBMClassifier(**params)
final_cla.fit(X, y)

cla_f1 = []
for tr_idx, va_idx in splits:
    cla = LGBMClassifier(**params)
    cla.fit(X.iloc[tr_idx], y[tr_idx])
    yhat_val = cla.predict(X.iloc[va_idx])
    cla_f1.append(f1_score(y[va_idx], yhat_val, average="macro"))

cv_macro_f1 = float(np.mean(cla_f1))

print(f"Grouped-CV macro-F1 (CLASSIFIER): {cv_macro_f1:.3f}")

bundle = {
    "model": final_cla,
    "metadata": {
        "best_params": params,
        "feature_order": FEATURES,
        "ordinal_map": {"bad":0, "moderate":1, "good":2},
        "cv_macro_f1": float(cv_macro_f1),
        "trained_on": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
}

joblib.dump(bundle, "models/LGBM_Posture_Classifier_v1.pkl")