import cv2 as cv
import time
import csv
from datetime import datetime
import mediapipe as mp
import math
import speech_recognition as sr
import threading
import numpy as np
import pygame
import os
import time
from os.path import exists
from urllib.request import urlretrieve

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

pygame.init()
pygame.mixer.init()

beep = pygame.mixer.Sound("bad_posture_alert.wav")

from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.pose import PoseLandmark

pose_model = "pose_landmarker_full.task"
if not exists(pose_model):
    pose_model = urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", "pose_landmarker_full.task")[0]

face_model = "face_landmarker.task"
if not exists(face_model):
    face_model = urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", "face_landmarker.task")[0]


last_log_time = time.time()
log_file = open("posture_trend_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])

FACE_LANDMARKS = {
    "left_eye_center": 159,   # Left eye center
    "right_eye_center": 386,  # Right eye center
    "nose_tip": 1,            # Nose tip
    "nose_bridge": 6,         # Nose bridge
    "chin": 175,              # Chin center
    "forehead": 10,           # Forehead center
    "left_cheek": 234,        # Left cheek
    "right_cheek": 454,       # Right cheek
    "left_temple": 127,       # Left temple
    "right_temple": 356,       # Right temple
}

POSE_LANDMARKS = {
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_hip": PoseLandmark.LEFT_HIP,
    "right_hip": PoseLandmark.RIGHT_HIP,
    "left_ear": PoseLandmark.LEFT_EAR,
    "right_ear": PoseLandmark.RIGHT_EAR,
}

ADDITIONAL_METRICS = {
    "face": ("left_eye_center", "right_eye_center", "nose_tip", "chin"),
    "hip": ("left_hip", "right_hip"),
    "clavicle": ("left_shoulder", "right_shoulder"),
    "torso": ("clavicle", "hip")
}

is_calibrating = False
calibration_data = {
    "facial_distances": [],
    "clavicle_distances": [],
    "clavicle_lengths": [],
    "shoulder_ear_dists": [],
    "slouch_angles": []
}
calibration_data_features = {}
previous_profile = {}

countdown_duration = 3
hold_duration = 5
calibrated_thresholds = {}
calibration_start_time = 0
bad_posture_start_time = None  # Variable that will hold how long someone has had bad posture
alert_active = False  # Checks is the alert is activated

mode = "front"

posture_status_labels = (
    ("Good Posture", (0, 255, 0)),
    ("Moderately Bad Posture", (255, 165, 0)),
    ("Bad Posture", (255, 0, 0))
)

#Globals for posture tracking
start_time = None
loop_started = False
last_beep_time = 0

drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
drawing_connections = mp.solutions.pose.POSE_CONNECTIONS
default_face_connections_style = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()


def listen_for_speech():
    global is_calibrating, calibration_data, calibration_start_time, mode, countdown_duration, hold_duration
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("[SpeechRecognition] Listening...")
            while True:
                try:
                    print("[DEBUG] Waiting for audio...")
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    print("[DEBUG] Audio captured")
                    command = recognizer.recognize_google(audio).lower().strip()
                    print(f"[SpeechRecognition] Heard: {command}")
                    if "calibrate" in command:
                        calibration_start_time = time.time()
                        is_calibrating = True
                        calibration_data = {k: [] for k in calibration_data}
                        print("Calibration countdown started. Get ready...")
                    elif "exit" in command:
                        print("[SpeechRecognition] Escape command received. Exiting program...")
                        import os
                        os._exit(0)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    print("[SpeechRecognition] API unavailable")
                    break
    except Exception as e:
        print("[ERROR] Microphone failed:", e)


print("[Thread] Starting voice command thread...")
threading.Thread(target=listen_for_speech, daemon=True).start()

def profile_from_landmarks(pose_landmarks, face_landmarks):
    global previous_profile
    profile = {}
    if not pose_landmarks or not face_landmarks:
        return profile

    for key, value in FACE_LANDMARKS.items():
        profile[key] = face_landmarks[0][value]
    for key, value in POSE_LANDMARKS.items():
        profile[key] = pose_landmarks[0][value]

    if previous_profile:
        for key, value in previous_profile.items():
            if profile.get(key):
                landmark = profile[key]
                profile[key] = type(landmark)(x=landmark.x, y=landmark.y, z=smooth(previous_profile[key].z, landmark.z, 0.2), visibility=landmark.visibility)

    for key, value in ADDITIONAL_METRICS.items():
        attrs = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
        for attr in attrs:
            attrs[attr] = sum(getattr(profile[landmark], attr) for landmark in value) / len(value)
        profile[key] = type(profile[value[0]])(x=attrs["x"], y=attrs["y"], z=attrs["z"], visibility=attrs["visibility"])

    previous_profile = profile
    return profile

def smooth(prev, current, alpha=0.2):
    return (1 - alpha) * prev + alpha * current

def gaussian_weight(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def scale_metric(value, low_threshold, high_threshold, max_score=3):
    """
    Scale a metric value to a 0-max_score range based on thresholds.

    Args:
        value: The metric value to scale
        low_threshold: Below this = score 0
        high_threshold: Above this = max_score
        max_score: Maximum score to return

    Returns:
        Integer score from 0 to max_score
    """
    if abs(value) < low_threshold:
        return 0
    elif abs(value) > high_threshold:
        return max_score
    else:
        # Linear interpolation between thresholds
        ratio = (abs(value) - low_threshold) / (high_threshold - low_threshold)
        return int(ratio * max_score)

def normalize_lighting(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv.merge((cl, a, b))
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

def draw_landmarks(frame, landmarks_list, connections, landmark_style):
    for i in range(len(landmarks_list)):
        landmarks = landmarks_list[i]

        # Draw the pose landmarks
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
        ])
        drawing_utils.draw_landmarks(
            frame,
            landmarks_proto,
            connections,
            landmark_style)

def landmark_dist_2d(a, b):
    return np.linalg.norm(np.array((a.x, a.y)) - np.array((b.x, b.y))).astype(float)

def analyze_posture(image, profile):
    global is_calibrating, calibration_start_time, countdown_duration, hold_duration, mode, calibration_data, calibrated_thresholds

    status = "No pose detected"
    average_color = image[30:310, 175:220].mean((0, 1))
    color = ((255 - average_color[0]), (255 - average_color[1]), (255 - average_color[2]))

    if not calibration_start_time:
        cv.putText(image, "Press 'c' to begin calibration for posture analysis", (30, 45), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return

    if not profile:
        return

    clavicle_length = landmark_dist_2d(profile["left_shoulder"], profile["right_shoulder"])
    facial_distance = profile["nose_tip"].z
    clavicle_distance = profile["clavicle"].z

    shoulder_ear_dist = (landmark_dist_2d(profile["left_shoulder"], profile["forehead"]) + landmark_dist_2d(profile["right_shoulder"], profile["forehead"])) / 2

    v1 = [profile["clavicle"].x - profile["nose_tip"].x, profile["clavicle"].y - profile["nose_tip"].y,
          profile["clavicle"].z - profile["nose_tip"].z]
    v2 = [profile["clavicle"].x - profile["torso"].x, profile["clavicle"].y - profile["torso"].y,
          profile["clavicle"].z - profile["torso"].z]
    slouch_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1).astype(float) * np.linalg.norm(v2).astype(float)))

    if is_calibrating:
        elapsed = time.time() - calibration_start_time
        if elapsed < countdown_duration:
            remaining = int(countdown_duration - elapsed) + 1
            cv.putText(image, f"Starting in: {remaining}s", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        elif elapsed < countdown_duration + hold_duration:
            cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
            calibration_data["facial_distances"].append(facial_distance)
            calibration_data["clavicle_distances"].append(clavicle_distance)
            calibration_data["clavicle_lengths"].append(clavicle_length)
            calibration_data["slouch_angles"].append(slouch_angle)
            calibration_data["shoulder_ear_dists"].append(shoulder_ear_dist)
        else:
            is_calibrating = False

            for key, value in calibration_data.items():
                calibration_data_features[key] = {"avg": np.mean(value), "std": np.std(value)}

            calibrated_thresholds = {
                "clavicle_length_threshold": calibration_data_features["clavicle_lengths"]["avg"] * 0.7,
            }
    else:
        mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"
        score = 0
        status = posture_status_labels[0]

        slouch_avg = calibration_data_features["slouch_angles"]["avg"]

        if mode == "front":
            facial_avg = calibration_data_features["facial_distances"]["avg"]
            clavicle_avg = calibration_data_features["clavicle_distances"]["avg"]
            shoulder_ear_avg = calibration_data_features["shoulder_ear_dists"]["avg"]


            facial_diff = (facial_distance - facial_avg)
            clavicle_diff = (clavicle_distance - clavicle_avg)
            facial_std = calibration_data_features["facial_distances"]["std"]
            clavicle_std = calibration_data_features["clavicle_distances"]["std"]


            slouch_metric = abs(slouch_angle - slouch_avg) * (1 - gaussian_weight(clavicle_diff, clavicle_avg, clavicle_std))
            head_metric = abs(facial_diff - clavicle_diff) * (1 + gaussian_weight(facial_diff, facial_avg, facial_std * 3) + gaussian_weight(clavicle_diff, clavicle_avg, clavicle_std * 3))

            score = min(2, scale_metric(slouch_metric, 0, 0.25, 2) + scale_metric(head_metric, 0, 0.3, 2))
        elif mode == "side":
            print(slouch_angle)

        status = posture_status_labels[score]
        cv.rectangle(image, (30, 180), (30 + score * 70, 200),
                     (0, 255 - score * 50, 50),
                     -1)
        cv.putText(image, f"Front Confidence: {score}", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   color, 1)
        cv.putText(image, status[0], (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, status[1], 2)

base_pose_options = python.BaseOptions(model_asset_path=pose_model)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_pose_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_tracking_confidence=0.7)

base_face_options = python.BaseOptions(model_asset_path=face_model)
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_face_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_face_detection_confidence=0.7,
    min_tracking_confidence=0.7)

cap = cv.VideoCapture(0)
cv.namedWindow('Posture Detection')

with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
    with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_rgb = cv.cvtColor(normalize_lighting(frame), cv.COLOR_BGR2RGB)
            timestamp = int(round(time.time() * 1000))
            pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
            face_results = face_landmarker.detect_for_video(face_image, timestamp)
            annotated_image = np.copy(frame)

            draw_landmarks(annotated_image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, drawing_styles.get_default_pose_landmarks_style())
            #draw_landmarks(annotated_image, face_results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, drawing_styles.DrawingSpec((255, 255, 255), 1, 1))

            analyze_posture(annotated_image, profile_from_landmarks(pose_results.pose_landmarks, face_results.face_landmarks))

            cv.imshow('Posture Detection', annotated_image)


            key = cv.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                calibration_start_time = time.time()
                is_calibrating = True
                calibration_data = {k: [] for k in calibration_data}

cap.release()
cv.destroyAllWindows()
log_file.close()
