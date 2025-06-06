import cv2 as cv
import time
import csv
from datetime import datetime
import mediapipe as mp
import math
import speech_recognition as sr
import threading
import numpy as np
from os.path import exists
from urllib.request import urlretrieve

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

is_calibrating = False
calibration_data = {
    "facial_distances": [],
    "torso_distances": [],
    "clavicle_lengths": [],
    "face_torso_heights": [],
    "shoulder_ear_distance": [],
    "clavicle_y": []
}
countdown_duration = 3
hold_duration = 5
calibrated_thresholds = {}
calibration_start_time = 0

mode = "front"
prev_facial_distances = [0, 0, 0, 0, 0, 0]
prev_torso_distances = [0, 0, 0, 0, 0, 0]
cache_idx = 0

status_enum = (
    ("Good Posture", (0, 220, 0)),
    ("Early Posture Warning", (0, 220, 0)),
    ("Significant Postural Issue", (0, 220, 0)),
    ("Severe Slouch + Lean", (0, 0, 255)),
    ("Critical Forward Posture", (128, 0, 0))
)

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


def calculate_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    angle_rad = math.acos(dot / (mag1 * mag2))
    return math.degrees(angle_rad)


def smooth(prev, current, alpha=0.2):
    return (1 - alpha) * prev + alpha * current


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

def analyze_posture(image, pose_landmarks):
    global is_calibrating, calibration_data, calibrated_thresholds, calibration_start_time, mode, countdown_duration, hold_duration, prev_facial_distances, prev_torso_distances, cache_idx, status_enum

    side_label = ""
    status = "No pose detected"
    color = (128, 128, 128)

    head_confidence_score = 0
    body_confidence_score = 0
    slouch_angle = 0

    left_shoulder = pose_landmarks[PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[PoseLandmark.RIGHT_SHOULDER]
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return  # Skip frame if landmarks are too uncertain
    left_hip = pose_landmarks[PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[PoseLandmark.RIGHT_HIP]
    left_ear = pose_landmarks[PoseLandmark.LEFT_EAR]
    right_ear = pose_landmarks[PoseLandmark.RIGHT_EAR]

    nose = pose_landmarks[PoseLandmark.NOSE]
    left_eye = pose_landmarks[PoseLandmark.LEFT_EYE]
    right_eye = pose_landmarks[PoseLandmark.RIGHT_EYE]
    mouth_left = pose_landmarks[PoseLandmark.MOUTH_LEFT]
    mouth_right = pose_landmarks[PoseLandmark.MOUTH_RIGHT]
    mouth = type(mouth_left)(x=(mouth_left.x + mouth_right.x) / 2,
                             y=(mouth_left.y + mouth_right.y) / 2,
                             z=(mouth_left.z + mouth_right.z) / 2,
                             visibility=1.0)

    face = type(mouth)(x=(left_eye.x + right_eye.x + nose.x + mouth.x) / 4,
                       y=(left_eye.y + right_eye.y + nose.y + mouth.y) / 4,
                       z=(left_eye.z + right_eye.z + nose.z + mouth.z) / 4,
                       visibility=1.0)

    clavicle = type(mouth)(x=(left_shoulder.x + right_shoulder.x) / 2,
                           y=(left_shoulder.y + right_shoulder.y) / 2,
                           z=(left_shoulder.z + right_shoulder.z) / 2,
                           visibility=1.0)

    hip = type(mouth)(x=(left_hip.x + right_hip.x) / 2,
                      y=(left_hip.y + right_hip.y) / 2,
                      z=(left_hip.z + right_hip.z) / 2,
                      visibility=1.0)

    torso = type(mouth)(x=(clavicle.x + hip.x) / 2,
                        y=(clavicle.y + hip.y) / 2,
                        z=(clavicle.z + hip.z) / 2,
                        visibility=1.0)

    prev_torso_distances[cache_idx] = torso.z
    prev_facial_distances[cache_idx] = face.z
    cache_idx = (cache_idx + 1) % 6

    torso_distance = np.mean(prev_torso_distances).astype(float)
    facial_distance = np.mean(prev_facial_distances).astype(float)
    head_tilt_difference = left_ear.y - right_ear.y

    clavicle_length = np.linalg.norm(
        np.array((left_shoulder.x, left_shoulder.y)) - np.array((right_shoulder.x, right_shoulder.y)))
    
    left_shoulder_ear = np.linalg.norm(np.array((left_shoulder.x, left_shoulder.y))- np.array ((left_ear.x, left_ear.y)))
    right_shoulder_ear = np.linalg.norm(np.array((right_shoulder.x, right_shoulder.y)) - np.array((right_ear.x, right_ear.y)))
    avg_shoulder_ear = (left_shoulder_ear + right_shoulder_ear) / 2

    face_torso_height = face.y - torso.y

    avg = lambda k: sum(calibration_data[k]) / len(calibration_data[k]) if calibration_data[k] else 0

    if is_calibrating:
        elapsed = time.time() - calibration_start_time
        if elapsed < countdown_duration:
            remaining = int(countdown_duration - elapsed) + 1
            cv.putText(image, f"Starting in: {remaining}s", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        elif elapsed < countdown_duration + hold_duration:
            calibration_data["torso_distances"].append(torso_distance)
            calibration_data["facial_distances"].append(facial_distance)
            calibration_data["clavicle_lengths"].append(clavicle_length)
            calibration_data["face_torso_heights"].append(face_torso_height)
            calibration_data["shoulder_ear_distance"].append(avg_shoulder_ear)
            calibration_data["clavicle_y"].append(clavicle.y)
            cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        else:
            is_calibrating = False
            calibrated_thresholds = {
                "clavicle_length_threshold": avg("clavicle_lengths") * 0.7
            }

    if calibrated_thresholds and not is_calibrating:
        mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"

        shoulder_ear_avg = avg("shoulder_ear_distance")
        shoulder_ear_percentage = (avg_shoulder_ear - shoulder_ear_avg) / shoulder_ear_avg
        status_idx = 0

        side_confidence_score = 0
        head_confidence_score = 0
        body_confidence_score = 0

        if mode == "front":
            facial_avg = avg("facial_distances")
            torso_avg = avg("torso_distances")
            face_clav_height_avg = avg("face_torso_heights")

            facial_percentage = (facial_distance - facial_avg) / facial_avg
            torso_percentage = (torso_distance - torso_avg) / torso_avg
            height_percentage = (face_torso_height - face_clav_height_avg) / face_clav_height_avg

            clavicle_y_current = clavicle.y
            clavicle_y_baseline = avg("clavicle_y")
            clavicle_y_change = clavicle_y_current - clavicle_y_baseline
            clavicle_y_pct = clavicle_y_change / clavicle_y_baseline if clavicle_y_baseline != 0 else 0
            if( clavicle_y_pct > 0.03):
                body_confidence_score += 3


            head_confidence_score += math.floor(abs(facial_percentage) / 0.15)
            head_confidence_score += math.floor(abs(head_tilt_difference) / 0.075)
            head_confidence_score = min(7, max(head_confidence_score, 0))

            body_confidence_score += math.floor(abs(facial_percentage - torso_percentage) / 0.125)
            body_confidence_score += math.floor(abs(height_percentage) / 0.1)
            body_confidence_score += math.floor(abs(shoulder_ear_percentage)/0.1)
            body_confidence_score = min(7, max(body_confidence_score, 0))

            combined_confidence = math.floor((head_confidence_score + body_confidence_score) / 2) - 1
            if combined_confidence < 1:
                combined_confidence = 0
            if combined_confidence > 4:
                combined_confidence = 4

            status_idx = status_enum[combined_confidence]

        elif mode == "side":

            # Determine which shoulder is closer to screen center
            facing = "left" if left_shoulder.z > right_shoulder.z else "right"
            if facing == "left":
                shoulder = right_shoulder
                hip = right_hip
                side_label = "Right Side View"
            else:
                shoulder = left_shoulder
                hip = left_hip
                side_label = "Left Side View"

            slouch_percentage = min(0, shoulder_ear_percentage * 100)
            side_confidence_score = min(7, max(math.floor(abs(slouch_percentage)), 0)) - 1
            if side_confidence_score < 1:
                side_confidence_score = 0
            if side_confidence_score > 3:
                side_confidence_score = 3

            status_idx = status_enum[side_confidence_score]

        status = status_idx[0]
        color = status_idx[1]


    average_color = frame[30:310, 175:220].mean((0, 1))
    final_color = ((255 - average_color[0]), (255 - average_color[1]), (255 - average_color[2]))

    cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (final_color[0], 220, final_color[2]), 2)

    if mode == "side":
        cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
        cv.rectangle(image, (30, 180), (30 + side_confidence_score * 40, 200),
                     (0, 255 - side_confidence_score * 50, 50),
                     -1)
        cv.putText(image, f"Slouch Confidence: {side_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   final_color, 1)
    else:
        cv.rectangle(image, (30, 180), (30 + head_confidence_score * 40, 200),
                     (0, 255 - head_confidence_score * 50, 50),
                     -1)
        cv.putText(image, f"Head Confidence: {head_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   final_color, 1)
        cv.rectangle(image, (30, 225), (30 + body_confidence_score * 40, 245),
                     (0, 255 - body_confidence_score * 50, 50),
                     -1)
        cv.putText(image, f"Body Confidence: {body_confidence_score}/7", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   final_color, 1)

    cv.putText(image, status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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

            timestamp = int(round(time.time() * 1000))
            pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
            face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

            pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
            face_results = face_landmarker.detect_for_video(face_image, timestamp)
            annotated_image = np.copy(frame)

            draw_landmarks(annotated_image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, drawing_styles.get_default_pose_landmarks_style())
            #draw_landmarks(annotated_image, face_results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, drawing_styles.DrawingSpec((255, 255, 255), 1, 1))

            if pose_results.pose_landmarks:
                analyze_posture(annotated_image, pose_results.pose_landmarks[0])

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
