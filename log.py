import csv, os, uuid, json, queue, threading, time, joblib, numpy as np

import pandas as pd
from scipy.special import softmax

bundle = joblib.load("models/posture_lgbm_classifier.pkl")
classifier_model = bundle["model"]
classifier_calibrator = bundle.get("calibrator")
classifier_decision_params = bundle.get("decision_params")
classifier_features = bundle.get("feature_names")
classifier_id_map = bundle["id_to_label"]


STATE_PATH = ".spinewise_state.json"
METRICS = ["head_tilt", "clavicle_drop_pct", "face_lean", "shoulder_ear_pct", "torso_lean_pct", "looking_down_pct"]

WINDOW_SIZE = 30 # in frames
WINDOW_STRIDE = 15 # how much to move window forward, 15 means reuse half of last window

LOG_SIZE = 6 # how many windows to log at a time
LOG_STRIDE = 2 # how much to move forward when logging batch of windows, 2 means log every other window

window_queue = queue.Queue(maxsize=WINDOW_SIZE)
window = []

io_tasks = set()

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

def retrieve_user_id():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))['user_id']
    user_id = str(uuid.uuid4())
    json.dump({'user_id': user_id}, open(STATE_PATH, 'w'))
    return user_id

def retrieve_session_id():
    return str(uuid.uuid4())

user_id = retrieve_user_id()
session_id = retrieve_session_id()

def record_frame(frame_data):
    if window_queue.full():
        return
    global window

    frame_data["ts"] = time.time()

    window.append(frame_data)
    if len(window) == WINDOW_SIZE:
        window_queue.put(pd.DataFrame(window))
        window = window[-WINDOW_STRIDE:]

# data logger
def log_posture_windows(filename="posture_dataset.csv"):
    with open(filename, "a", newline="") as new_file:
        writer = csv.writer(new_file)
        windows = []

        while True:
            while len(windows) < LOG_SIZE:
                windows.append(window_queue.get())

            # components for simple linear regression through minimizing sum of squared residuals
            x_axis = np.arange(len(windows[0]))
            x_cov = (x_axis - x_axis.mean())
            denom = ((x_axis - x_axis.mean()) ** 2).sum()

            rows = []

            for w in range(0, len(windows), LOG_STRIDE):
                features = {"user_id": user_id, "session_id": session_id}
                timestamps = windows[w]["ts"]

                for metric in METRICS:
                    window_data = windows[w][metric].to_numpy()
                    max = window_data.max()
                    min = window_data.min()

                    features[f"{metric}_mean"] = window_data.mean()
                    features[f"{metric}_std"] = window_data.std()
                    features[f"{metric}_min"] = min
                    features[f"{metric}_max"] = max
                    features[f"{metric}_range"] = max - min

                    y_cov = (window_data - window_data.mean())
                    least_squares_slope = (x_cov * y_cov).sum() / denom if denom > 0.0 else 0.
                    features[f"{metric}_slope"] = least_squares_slope
                    features[f"{metric}_delta"] = window_data[-1] - window_data[0]
                features["t_start"] = timestamps.iloc[0]
                features["t_end"] = timestamps.iloc[-1]
                features["label"] = windows[w]["label"].mode().iloc[0]
                rows.append(features)

                trimmed_features = pd.DataFrame([features])[classifier_features]
                logits = classifier_model.predict(trimmed_features, raw_score=True)
                cal_logits = classifier_calibrator.decision_function(logits)
                yhat = apply_decision_layer(classifier_decision_params, cal_logits)

                print(f"[DEBUG] classifier result: {classifier_id_map[yhat[0]]}")

                if new_file.tell() == 0:
                    writer.writerow(features.keys())
            for i in range(0, len(rows)):
                writer.writerow(rows[i].values())

            windows.clear()

threading.Thread(target=log_posture_windows, daemon=True).start()