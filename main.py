import cv2 as cv
import time
import csv
from datetime import datetime
import mediapipe as mp
import math
import speech_recognition as sr
import threading
import numpy as np

last_log_time = time.time()
log_file = open("posture_trend_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

is_calibrating = False
calibration_data = {
    "facial_distances": [],
    "clavicle_distances": [],
    "clavicle_lengths": [],
    "face_clavicle_heights": []
}
countdown_duration = 3
hold_duration = 5
calibrated_thresholds = {}

mode = "front"


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


cap = cv.VideoCapture(0)
cv.namedWindow('Posture Detection')

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    prev_slouch_angle = 0
    prev_z_diff_nose = 0
    prev_spine_angle = 0

    current_status = "No pose detected"
    last_status_change_time = time.time()
    displayed_status = "No pose detected"
    displayed_color = (128, 128, 128)

    while cap.isOpened():
        side_label = ""  # default in case we're in front mode
        head_tilt_status = None  # define early so it's always available
        shoulder_leaning_forward = False  # Initialize to prevent crash

        head_confidence_score = 0
        body_confidence_score = 0

        success, frame = cap.read()
        if not success:
            continue

        frame = normalize_lighting(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.pose_landmarks and results.pose_world_landmarks:
            new_status = current_status
            new_color = displayed_color

            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark
            # Pose confidence filter

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                continue  # Skip frame if landmarks are too uncertain
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
            mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
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

            clavicle_distance = clavicle.z
            facial_distance = face.z
            head_tilt_difference = left_ear.y - right_ear.y

            clavicle_length = np.linalg.norm(
                np.array((left_shoulder.x, left_shoulder.y)) - np.array((right_shoulder.x, right_shoulder.y)))

            face_clavicle_height = face.y - clavicle.y

            if is_calibrating:
                elapsed = time.time() - calibration_start_time
                if elapsed < countdown_duration:
                    remaining = int(countdown_duration - elapsed) + 1
                    cv.putText(image, f"Starting in: {remaining}s", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 255), 2)
                elif elapsed < countdown_duration + hold_duration:
                    calibration_data["clavicle_distances"].append(clavicle_distance)
                    calibration_data["facial_distances"].append(facial_distance)
                    calibration_data["clavicle_lengths"].append(clavicle_length)
                    calibration_data["face_clavicle_heights"].append(face_clavicle_height)
                    cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 255), 2)
                else:
                    is_calibrating = False
                    avg = lambda k: sum(calibration_data[k]) / len(calibration_data[k]) if calibration_data[k] else 0
                    calibrated_thresholds = {
                        "clavicle_length_threshold": avg("clavicle_lengths") * 0.6
                    }

            if calibrated_thresholds and not is_calibrating:
                mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"
                if mode == "front":
                    facial_avg = avg("facial_distances")
                    clav_avg = avg("clavicle_distances")
                    face_clav_height_avg = avg("face_clavicle_heights")

                    facial_percentage = (facial_distance - facial_avg) / facial_avg
                    clav_percentage = (clavicle_distance - clav_avg) / clav_avg
                    height_percentage = (face_clavicle_height - face_clav_height_avg) / face_clav_height_avg

                    head_confidence_score += math.floor(abs(facial_percentage) / 0.1)
                    head_confidence_score += math.floor(abs(head_tilt_difference) / 0.05)
                    head_confidence_score = min(7, max(head_confidence_score, 0))

                    body_confidence_score += math.floor(abs(facial_percentage - clav_percentage) / 0.1)
                    body_confidence_score += math.floor(abs(height_percentage) / 0.1)
                    body_confidence_score = min(7, max(body_confidence_score, 0))
                elif mode == "side":
                    pass

            if new_status != current_status:
                current_status = new_status
                last_status_change_time = time.time()
            elif time.time() - last_status_change_time > 1:
                displayed_status = current_status
                displayed_color = new_color

                if time.time() - last_log_time >= 10:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if mode == "front":
                        facing = "front"
                    csv_writer.writerow([
                        timestamp,
                        mode,
                        facing,
                        current_status,
                        head_tilt_status if head_tilt_status else "neutral",
                        head_confidence_score,
                        body_confidence_score
                    ])
                    log_file.flush()
                    last_log_time = time.time()

            average_color = frame[30:310, 175:220].mean((0, 1))
            negative_color = (255 - average_color[0], 255 - average_color[1], 255 - average_color[2])

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, negative_color, 2)
            cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
            #cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, displayed_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, displayed_color, 2)
            #if head_tilt_status:
            #cv.putText(image, head_tilt_status, (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            #if shoulder_leaning_forward:
            #cv.putText(image, "Leaning In", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            cv.rectangle(image, (30, 180), (30 + head_confidence_score * 40, 200), (0, 255 - head_confidence_score * 50, 50), -1)
            cv.putText(image, f"Head Confidence: {head_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       negative_color, 1)
            cv.rectangle(image, (30, 225), (30 + body_confidence_score * 40, 245), (0, 255 - body_confidence_score * 50, 50), -1)
            cv.putText(image, f"Body Confidence: {body_confidence_score}/7", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       negative_color, 1)

        cv.imshow('Posture Detection', image)
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
