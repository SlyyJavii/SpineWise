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

good_values = df[df["label"] == "good"]

dict1 = {}
dict2 = {}

for f in FEATURES:
    col = good_values[f].to_numpy()
    dict1[f] = np.median(col)
    dict2[f] = np.percentile(col, 75) - np.percentile(col, 25)

bundle = joblib.load("models/posture_lgbm_classifier.pkl")

bundle["dataset_baseline_median"] = dict1
bundle["dataset_baseline_iqr"] = dict2

jj = {0: "hi", 1: "hello", 2: "ok"}

#joblib.dump(bundle, "models/posture_lgbm_classifier.pkl")

#bundle_path = "models/posture_lgbm_classifier.pkl"
#joblib.dump(
    #{"model": best_clf, "feature_names": FEATURES, "label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL, "calibrator": calibrator, "decision_params": best_decision_params},
    #bundle_path
#)