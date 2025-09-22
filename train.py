import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CSV = "posture_dataset.csv"

# Metrics per frame/window
METRICS = ["head_tilt","clavicle_drop_pct","face_lean","shoulder_ear_pct","torso_lean_pct","looking_down_pct"]

# Stats per window
STATS = ["mean","std","min","max","range","slope","delta"]

def build_feature_list(metrics=METRICS, stats=STATS):
    return [f"{m}_{s}" for m in metrics for s in stats]

df = pd.read_csv(CSV)
FEATURES = build_feature_list()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected feature columns: {missing}")

# keep metadata for grouping/QA; exclude from X when fitting
NON_FEATURES = ["user_id","session_id","t_start","t_end","label"]
for c in NON_FEATURES:
    if c not in df.columns: df[c] = np.nan

X = df[FEATURES]
y_str = df["label"].astype(str)
#groups = df["user_id"].astype(str)

ORDINAL_MAP = {"bad":0, "moderate":1, "good":2}
y_ord = y_str.map(ORDINAL_MAP).astype(int).to_numpy()

#cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) # the preferred cross-validator, only works if dataset has more than 1 user
#splits = list(cv.split(X, y_ord, groups=groups))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # regular cross-validator, create 5 folds from dataset each with train and test sets that have a similar class composition to the entire dataset
splits = list(cv.split(X, y_ord))

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
    verbosity=-1
)

param_dist = { # generate range of possible parameters for LightGBM
    "learning_rate": np.geomspace(0.05, 0.10, 6),
    "n_estimators": np.linspace(600, 1000, 9, dtype=int),
    "max_depth": [-1, 3, 4, 5],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 40],
    "min_child_weight": np.geomspace(1e-3, 1.0, 5),
    "min_split_gain": np.linspace(0, 0.1, 5, dtype=int),
    "lambda_l1": np.geomspace(1e-3, 0.1, 6),
    "lambda_l2": np.geomspace(0.3, 3.0, 6),
    "feature_fraction": [0.8, 0.9, 1.0],
    "bagging_fraction": [0.8, 0.9, 1.0],
    "bagging_freq": [0, 1, 2],
}

search = RandomizedSearchCV( # random search that will yield a model with the best performing parameters. recommendation: look into different search techniques like gridsearch
    estimator=reg0,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_mean_squared_error",  # regression metric for search
    cv=splits,
    n_jobs=-1,
    random_state=42,
    refit=True,
    verbose=1
)
search.fit(X, y_ord)
best_reg = search.best_estimator_

# because model is a regressor with a continuous output value between 0 and 2, find the best t0/t1 thresholds that define bad/moderate/good
def apply_thresholds(y_hat, t1, t2):
    return np.where(y_hat <= t1, 0, np.where(y_hat >= t2, 1, 2))

def tune_thresholds(y_true, y_hat):
    t1_grid = np.linspace(0.1, 1.2, 45)
    t2_grid = np.linspace(0.9, 1.9, 45)
    best = (-1, 0.5, 1.5)

    for t1 in t1_grid:
        for t2 in t2_grid:
            if t1 > t2:
                continue
            f1 = f1_score(y_true, apply_thresholds(y_hat, t1, t2), average="macro")
            if f1 > best[0]:
                best = (f1, t1, t2)
    return best

# cross-validated threshold selection. for each fold, use model w/ best parameters, train on training set, predict with test set, see which t1 and t2 led to the best f1 score
fold_f1, t1s, t2s = [], [], []
for train_index, test_index in splits:
    reg = LGBMRegressor(**best_reg.get_params())
    reg.fit(X.iloc[train_index], y_ord[train_index])
    y_hat = reg.predict(X.iloc[test_index])
    f1, t1, t2 = tune_thresholds(y_ord[test_index], y_hat)
    fold_f1.append(f1)
    t1s.append(t1)
    t2s.append(t2)

t1_cv, t2_cv = float(np.median(t1s)), float(np.median(t2s))
print(f"CV macro-F1 (ordinal): {np.mean(fold_f1):.3f}, t1={t1_cv:.3f}, t2={t2_cv:.3f}")