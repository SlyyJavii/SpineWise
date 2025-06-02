# This is the main.py of the (ongoing) SpineWise posture rating system application project for Capstone 1 and 2.
# This code utilizes MediaPipe and OpenCV to track landmarks on the human body and assess the quality of posture
# from both front and side views. This currently encompasses the base backend of the code.
# Honestly this documentation isn't good because it's already depreciated.
# Will be doing the same for jp-1c-posture-opt because it's up to date with latest MediaPipe.

import cv2 as cv
import time
import csv
from datetime import datetime
import mediapipe as mp
import math
import speech_recognition as sr
import threading
import numpy as np

# Open a CSV file that logs timestamp of latest entry.
# This generates a CSV with a timestamp, mode parameter, facing, posture status,
# head tilt, and confidence score. (Note: Juan has never opened these CSVs so I don't know how they look.)
last_log_time = time.time()
log_file = open("posture_trend_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])

# Basic MediaPipe posing and drawing calls.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calibration calls.
is_calibrating = False
calibration_data = {
    "facial_distances": [], # ASSUMING this is face-to-camera depth for angle calculation against torso distance.
    "torso_distances": [], # Ditto.
    "clavicle_lengths": [], # Shoulder-to-shoulder distance. This aids in the later mode-switch part of the code.
    "face_torso_heights": [], # Height from face-to-torso.

}
countdown_duration = 3 # Countdown.
hold_duration = 5 # Holds for 5 seconds; will most likely be configurable in the future since it gets annoying to sit for 5 seconds.
calibrated_thresholds = {} # Storage of calibration thresholds.

mode = "front" # Initial mode assumes user is facing front when program opens. Something to look into with new MP.

# This utilizes the microphone in the computer in a separate thread to listen for command words.
# Made for hands-free use of posture program when away from computer.
# Listens for "calibrate" or "exit" for their respective needs. Might need some sort of "ignore" and "listen"
# function down the line if you're using this while giving a speech so you don't ACCIDENTALLY kill the program.
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

# Math that DIRECTLY corresponds with calculating angles of slouch, tilts, etc.
def calculate_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    angle_rad = math.acos(dot / (mag1 * mag2))
    return math.degrees(angle_rad)

# what
# Assuming this involves the math for the code, reducing 'jitter' between values so they transition a lot easier in calculation.
def smooth(prev, current, alpha=0.2):
    return (1 - alpha) * prev + alpha * current

# Lighting adjust for high-light/low-light conditions, for better readings.
def normalize_lighting(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv.merge((cl, a, b))
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

# Opens up the webcam.
cap = cv.VideoCapture(0)
cv.namedWindow('Posture Detection')

# Another important part of code. MP tracks webcam frames.

# This makes a mean of 3 previous distances of their corresponding variables.
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    prev_torso_distances = [0, 0, 0]
    prev_facial_distances = [0, 0, 0]
    cache_idx = 0 # This indexes into circular buffer.

    # Text. Self-explanatory.
    current_status = "No pose detected"
    last_status_change_time = time.time()
    displayed_status = "No pose detected"
    displayed_color = (128, 128, 128)

    # Additional text and corresponding values.
    while cap.isOpened():
        side_label = ""  # default in case we're in front mode
        head_tilt_status = None  # define early so it's always available
        shoulder_leaning_forward = False  # Initialize to prevent crash

        head_confidence_score = 0 # Confidence score system that increases with discrepancies and caps out at 7.
        body_confidence_score = 0 # Ditto.

        success, frame = cap.read() # Simple frame-read. If frame isn't read, just read next frame.
        if not success:
            continue

        frame = normalize_lighting(frame) # Normalize lighting block from earlier.
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # BGR for OpenCV, RGB for MediaPipe.
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Landmark draws and status calls in relation to landmark draws.
        if results.pose_landmarks and results.pose_world_landmarks:
            new_status = current_status
            new_color = displayed_color

            # Landmark retrievals.
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark
            # Pose confidence filter

            # Frame skips for when shoulder landmarks are not detected well.
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                continue  # Skip frame if landmarks are too uncertain

            # Landmark extracts.
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
            mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

            # Mouth midpoint as average of corners.
            mouth = type(mouth_left)(x=(mouth_left.x + mouth_right.x) / 2,
                                     y=(mouth_left.y + mouth_right.y) / 2,
                                     z=(mouth_left.z + mouth_right.z) / 2,
                                     visibility=1.0)

            # Facial average of all features for later math. This is for facial distance.
            face = type(mouth)(x=(left_eye.x + right_eye.x + nose.x + mouth.x) / 4,
                                     y=(left_eye.y + right_eye.y + nose.y + mouth.y) / 4,
                                     z=(left_eye.z + right_eye.z + nose.z + mouth.z) / 4,
                                     visibility=1.0)

            # Clavicle made from shoulder-shoulder midpoint.
            clavicle = type(mouth)(x=(left_shoulder.x + right_shoulder.x) / 2,
                                     y=(left_shoulder.y + right_shoulder.y) / 2,
                                     z=(left_shoulder.z + right_shoulder.z) / 2,
                                     visibility=1.0)

            # Hip midpoint. Proves useful for far away but not upclose since hips won't be visible.
            hip = type(mouth)(x=(left_hip.x + right_hip.x) / 2,
                                     y=(left_hip.y + right_hip.y) / 2,
                                     z=(left_hip.z + right_hip.z) / 2,
                                     visibility=1.0)

            # Torso midpoint between clavicle and hip. Ditto like hip purpose.
            torso = type(mouth)(x=(clavicle.x + hip.x) / 2,
                                     y=(clavicle.y + hip.y) / 2,
                                     z=(clavicle.z + hip.z) / 2,
                                     visibility=1.0)

            # Average of previous measurements of these .z's.
            prev_torso_distances[cache_idx] = torso.z
            prev_facial_distances[cache_idx] = face.z
            cache_idx = (cache_idx + 1) % 3

            # Mean of above to reduce value jitter.
            torso_distance = np.mean(prev_torso_distances).astype(float)
            facial_distance = np.mean(prev_facial_distances).astype(float)
            head_tilt_difference = left_ear.y - right_ear.y # Head tilt computation using ear.y's.

            # Clavicle width calculations using normalized coords.
            clavicle_length = np.linalg.norm(
                np.array((left_shoulder.x, left_shoulder.y)) - np.array((right_shoulder.x, right_shoulder.y)))

            # Self-explanatory. Usage of .y coordinates for height calculation between face mid and torso mid.
            face_torso_height = face.y - torso.y

            # Calibration block. Grabs all values during a snapshot and stores for reference with later values.
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
                    cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 255), 2)
                else:
                    is_calibrating = False
                    avg = lambda k: sum(calibration_data[k]) / len(calibration_data[k]) if calibration_data[k] else 0
                    calibrated_thresholds = {
                        "clavicle_length_threshold": avg("clavicle_lengths") * 0.6
                    }

            # Once values are snapshotted, and NO more calibration is happening concurrently, we compare cali-vals with current vals.
            if calibrated_thresholds and not is_calibrating:
                # Automatic side mode calculation; if clavicle length is less than calibrated, switch to side.
                mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"
                if mode == "front": # Front mode scoring logic below. First three cover averages in comparison to calibration for scoring.
                    facial_avg = avg("facial_distances")
                    torso_avg = avg("torso_distances")
                    face_clav_height_avg = avg("face_torso_heights")

                    # Percent change relative to cali-avgs.
                    facial_percentage = (facial_distance - facial_avg) / facial_avg
                    torso_percentage = (torso_distance - torso_avg) / torso_avg
                    height_percentage = (face_torso_height - face_clav_height_avg) / face_clav_height_avg

                    # Confidence system. This operates by how far values deviate from the calibrated 'norm'. I.E.:
                    head_confidence_score += math.floor(abs(facial_percentage) / 0.2) # Face move from OG position. Every 20% deviation adds a point.
                    head_confidence_score += math.floor(abs(head_tilt_difference) / 0.075) # Head TILT from OG position. Every 7.5% deviation adds a point.
                    head_confidence_score = min(7, max(head_confidence_score, 0)) # Score between 0-7.

                    body_confidence_score += math.floor(abs(facial_percentage - torso_percentage) / 0.1) # Face vs torso depth change. +10% deviation = +1 point.
                    body_confidence_score += math.floor(abs(height_percentage) / 0.1) # Vertical height change between face and torso. +10% = +1p.
                    body_confidence_score = min(7, max(body_confidence_score, 0)) # Ditto score range from head.
                elif mode == "side":
                    pass

            # Updates status with 1 second buffer as anti-flicker.
            if new_status != current_status:
                current_status = new_status
                last_status_change_time = time.time()
            elif time.time() - last_status_change_time > 1:
                displayed_status = current_status
                displayed_color = new_color

                # CSV write dumps. Self-explanatory.
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

            # Colors for text. In grey for readability against tester's high-light environments.
            average_color = frame[30:310, 175:220].mean((0, 1))
            gray_negative = 0.3 * (255 - average_color[0]) +  0.59 * (255 - average_color[1]) + 0.11 * (255 - average_color[2])
            final_color = (gray_negative, gray_negative, gray_negative)

            # This is all the text labels that show up for debug. Mode, Head Confidence, Body Confidence,
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (gray_negative, 220, gray_negative), 2)
            cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
            #cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, displayed_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, displayed_color, 2)
            #if head_tilt_status:
            #cv.putText(image, head_tilt_status, (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            #if shoulder_leaning_forward:
            #cv.putText(image, "Leaning In", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            cv.rectangle(image, (30, 180), (30 + head_confidence_score * 40, 200), (0, 255 - head_confidence_score * 50, 50), -1)
            cv.putText(image, f"Head Confidence: {head_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       final_color, 1)
            cv.rectangle(image, (30, 225), (30 + body_confidence_score * 40, 245), (0, 255 - body_confidence_score * 50, 50), -1)
            cv.putText(image, f"Body Confidence: {body_confidence_score}/7", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       final_color, 1)

        # Keyboard inputs for user.
        cv.imshow('Posture Detection', image)
        key = cv.waitKey(5) & 0xFF
        if key == 27: # ESC to exit.
            break
        elif key == ord('c'): # (C) for calibration.
            calibration_start_time = time.time()
            is_calibrating = True
            calibration_data = {k: [] for k in calibration_data}

# Resource clean-up on exit.
cap.release() # Closes video file.
cv.destroyAllWindows() # Closes all windows.
log_file.close() # Close CSV log