import csv, os, uuid, json, queue, threading, time, numpy as np

import pandas as pd

STATE_PATH = ".spinewise_state.json"
METRICS = ["head_tilt", "clavicle_drop_pct", "face_lean", "shoulder_ear_pct", "torso_lean_pct", "looking_down_pct"]

WINDOW_SIZE = 30 # in frames
WINDOW_STRIDE = 15 # how much to move window forward, 15 means reuse half of last window

LOG_SIZE = 6 # how many windows to log at a time
LOG_STRIDE = 2 # how much to move forward when logging batch of windows, 2 means log every other window

window_queue = queue.Queue(maxsize=WINDOW_SIZE)
window = []

io_tasks = set()

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

                if new_file.tell() == 0:
                    writer.writerow(features.keys())
            for i in range(0, len(rows)):
                writer.writerow(rows[i].values())

            windows.clear()

threading.Thread(target=log_posture_windows, daemon=True).start()