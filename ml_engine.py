import math
import time, joblib, threading
from collections import defaultdict

from scipy.special import softmax as _softmax
import pandas as pd
import numpy as np
import shap
from scipy.stats import logistic

BUFFER_SIZE = 30

_ML = {
    "loaded": False,
    "model": None,
    "calibrator": None,
    "decision_params": None,
    "label_to_id": None,
    "id_to_label": None,
    "feature_names": None,
    "baseline_median": None,
    "baseline_iqr": None,
    "explainer": None,
    "window": [],
    "window_ts": 0.0
}

session_ctx = {
    "good_windows": pd.DataFrame(),
    "total_good_seen": 0,

    "last_label": None,
    "median": None,
    "iqr": None,
    "shap_values": None,
    "z_scores": None,
    "need_scores": None,
    "proba": None
}

PATTERNS = {
    "forward_head": {
        "features": ["face_lean", "shoulder_ear_pct", "looking_down_pct"],
    },
    "slouched_sitting": {
        "features": ["torso_lean_pct", "clavicle_drop_pct"],
    },
    "rounded_shoulders": {
        "features": ["clavicle_drop_pct", "shoulder_ear_pct"],
    },
    "lateral_head_tilt": {
        "features": ["head_tilt", "looking_down_pct"],
    }
}

window_mutex = threading.Lock()

# noinspection PyTypeChecker
def _load_ml_once(path = "models/posture_lgbm_classifier.pkl"):
    # load trained LGBM
    if _ML["loaded"]:
        return
    try:
        bundle = joblib.load(path)
        _ML["model"] = bundle.get("model")
        _ML["calibrator"] = bundle.get("calibrator")      # may be None
        _ML["decision_params"] = bundle.get("decision_params")  # may be None
        _ML["label_to_id"] = bundle.get("label_to_id")
        _ML["id_to_label"] = bundle.get("id_to_label")
        _ML["loaded"] = True
        _ML["explainer"] = shap.TreeExplainer(_ML["model"])
        _ML["baseline_median"] = pd.DataFrame([bundle.get("dataset_baseline_median")])
        _ML["baseline_iqr"] = pd.DataFrame([bundle.get("dataset_baseline_iqr")])
        _ML["feature_names"] = bundle.get("feature_names")
        _ML["window_ts"] = time.monotonic()
        print("[BACKEND] ML bundle loaded.")
    except Exception as e:
        print(f"[BACKEND] ML bundle load failed: {e}")
        _ML["loaded"] = False

def _apply_decision_layer(decision_params, logits):
    eps     = decision_params["eps"]
    alpha   = decision_params["alpha"]
    beta    = decision_params["beta"]
    tau     = decision_params["tau"]
    biasmod = decision_params["bias_mod"]

    new_logits = logits / tau
    new_logits[:, 1] += biasmod
    proba = _softmax(new_logits, axis=1)

    C = np.array([
        [0.0, 1 + alpha, 1.0],
        [beta, 0.0, beta],
        [1.0, 1 + alpha, 0.0]
    ])

    expected = proba @ C.T
    y_argmin = expected.argmin(1)
    session_ctx["proba"] = proba[0]

    close_to_top = (proba.max(1) - proba[:, 1] <= eps)
    return np.where(close_to_top, 1, y_argmin)

def analyze_window(window) -> pd.DataFrame:
    if len(window) == 0:
        return pd.DataFrame()

    base_window_df = pd.DataFrame(window)
    final_window = {}
    for metric in window[0].keys():
        if metric == "label":
            continue
        window_data = base_window_df[metric].to_numpy()

        window_min = window_data.min()
        window_max = window_data.max()

        x_axis = np.arange(len(window))
        x_cov = (x_axis - x_axis.mean())
        denom = (x_cov ** 2).sum()

        final_window[f"{metric}_mean"] = window_data.mean()
        final_window[f"{metric}_std"] = window_data.std()
        final_window[f"{metric}_min"] = window_min
        final_window[f"{metric}_max"] = window_max
        final_window[f"{metric}_range"] = window_max - window_min
        final_window[f"{metric}_slope"] = (x_cov *
                    (window_data - window_data.mean())).sum() / denom if denom > 0.0 else 0.0
        final_window[f"{metric}_delta"] = window_data[-1] - window_data[0]
    return pd.DataFrame([final_window])[_ML["feature_names"]]


def record_frame(features_dict):
    if not _ML["loaded"]:
        return False

    if (time.monotonic() - _ML["window_ts"]) < 1:
        _ML["window"].append(features_dict)
        return
    else:
        window_mutex.acquire()
        window_df = analyze_window(_ML["window"])
        _ML["window"] = []
        _ML["window_ts"] = time.monotonic()
        window_mutex.release()
        return window_df

def update_good_windows(good_windows, window_df, total_good):
    if good_windows is None:
        good_windows = window_df.copy()

    good_windows = pd.concat([good_windows, window_df], ignore_index=True)
    if len(good_windows) > BUFFER_SIZE:
        good_windows = good_windows.iloc[-BUFFER_SIZE:]
    total_good += 1
    return good_windows, total_good

def blend_baseline(global_med, global_iqr, good_windows, total_good, k=40):
    if good_windows is None or len(good_windows) < 5:
        return global_med, global_iqr

    personal_med = good_windows.median()
    personal_iqr = good_windows.quantile(0.75) - good_windows.quantile(0.25)

    lam = float(total_good) / (total_good + k)
    blended_med = (1 - lam) * global_med + lam * personal_med
    blended_iqr = (1 - lam) * global_iqr + lam * personal_iqr

    return blended_med, blended_iqr

def z_scores(window_df):
    if session_ctx["median"] is None or session_ctx["iqr"] is None:
        med = _ML["baseline_median"]
        iqr = _ML["baseline_iqr"]
    else:
        med = session_ctx["median"]
        iqr = session_ctx["iqr"]

    return (window_df - med) / (1e-6 + iqr / 1.349)


# one-step ML predictor from the current per-frame features
def ml_predict_label_from_features(features_dict) -> str:
    # use the trained model on the instant features you already compute per frame
    # if calibrator+decision layer exist, use them else fall back to proba argmax

    _load_ml_once()
    if not _ML["loaded"] or _ML["model"] is None:
        return features_dict.get("label", "Good Posture")  # fallback to current result

    window_df = record_frame(features_dict)
    if window_df is None:
        return session_ctx["last_label"]

    # Model was trained on many engineered columns (means/std/etc)
    # for live usage we approximate that by aggregating a few seconds of per-frame data (2 s smoothing window)
    # this provides a pragmatic bridge between instantaneous inputs and the model’s expected short-term statistics
    try:
        # prefer feature_names if present in bundle
        model = _ML["model"]
        calibrator = _ML.get("calibrator")
        decision_params = _ML.get("decision_params")

        if calibrator is not None and decision_params is not None:
            # Use raw_score -> calibrator -> decision layer
            logits = model.predict(window_df, raw_score=True)  # shape (1, 3)
            cal_logits = calibrator.decision_function(logits)  # (1, 3) for multinomial LR
            yhat = _apply_decision_layer(decision_params, cal_logits.reshape(1, -1))
            cls_id = int(yhat[0])
        else:
            # no calibrator, use probabilities
            logits = model.predict(window_df, raw_score=True)
            proba = _softmax(logits, axis=1)
            session_ctx["proba"] = proba[0]

            cls_id = int(np.argmax(proba, axis=1)[0])

        if cls_id == 2:
            session_ctx["good_windows"], session_ctx["total_good_seen"] = update_good_windows(session_ctx["good_windows"], window_df, session_ctx["total_good_seen"])
            session_ctx["median"], session_ctx["iqr"] = blend_baseline(_ML["baseline_median"], _ML["baseline_iqr"], session_ctx["good_windows"], session_ctx["total_good_seen"])

        session_ctx["shap_values"] = _ML["explainer"].shap_values(window_df)

        final_z_scores = defaultdict(float)
        for feature, z_score in z_scores(window_df).to_dict(orient="list").items():
            if type(feature) != str:
                break
            final_z_scores[get_feature_base(feature)] += z_score[0]

        session_ctx["z_scores"] = final_z_scores

        if cls_id < 2:
            pattern_shares = get_patterns_from_model(cls_id)
            need_scores = {}
            for pattern, cfg in PATTERNS.items():
                z_vals = [abs(final_z_scores[f]) for f in cfg["features"]]
                severity = max(z_vals) if z_vals else 0.0

                norm_severity = 1 / (1 + math.exp(-(severity - 1.5)))
                smoothed_share = (pattern_shares[pattern] ** 1.5)

                need_scores[pattern] = (norm_severity ** 0.4) * (smoothed_share ** 0.6)

            session_ctx["need_scores"] = need_scores

        # map id -> label string (bad/moderate/good)
        id2lab = {0: "Bad Posture", 1: "Moderate Posture", 2: "Good Posture"}
        session_ctx["last_label"] = id2lab[cls_id]
        return id2lab.get(cls_id, "Moderate Posture")
    except Exception as e:
        print(f"[BACKEND] ML live predict failed: {e}")
        return "Good Posture"

def get_feature_base(feature):
    parts = feature.split("_")
    return "_".join(parts[:-1])

def get_patterns_from_model(label_id):
    if session_ctx["shap_values"] is None or session_ctx["z_scores"] is None:
        return

    feature_shaps = defaultdict(float)
    for feature, values in zip(_ML["feature_names"], session_ctx["shap_values"][0]):
        base_feature = get_feature_base(feature)
        value = values[label_id]
        if value <= 0:
            continue

        feature_shaps[base_feature] += value

    pattern_shaps = {}
    for pattern, cfg in PATTERNS.items():
        total = sum(feature_shaps[feature] for feature in cfg["features"])
        pattern_shaps[pattern] = total

    total_shap = sum(pattern_shaps.values())
    if total_shap > 0:
        pattern_share = {pattern: value / total_shap for pattern, value in pattern_shaps.items()}
    else:
        pattern_share = {pattern: 0.0 for pattern in pattern_shaps}

    return pattern_share