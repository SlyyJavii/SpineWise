from colorsys import hsv_to_rgb

import cv2 as cv
import time
import mediapipe as mp
import math

import numpy as np
import speech_recognition as sr
import threading

# Initialize MediaPipe drawing and pose estimation modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calibration state and data containers
is_calibrating = False
countdown_duration = 3
hold_duration = 5
calibration_start_time = 0
calibration_targets = {"slouch_angles", "z_diffs", "spine_angles", "sitting_heights", "head_to_shoulder_heights", "clavicle_lengths"}
calibration_data = {k: [] for k in calibration_targets}
calibrated_avgs = {}

# Start speech recognition listener
def listen_for_speech():
    global is_calibrating, calibration_data, calibration_start_time, mode, countdown_duration, hold_duration
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("[SpeechRecognition] Listening...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower().strip()
                print(f"[SpeechRecognition] Heard: {command}")
                if "calibrate" in command:
                    calibration_start_time = time.time()
                    is_calibrating = True
                    print("Calibration countdown started. Get ready...")
                elif "switch" in command:
                    mode = "side" if mode == "front" else "front"
                    print("Switched to", mode)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("[SpeechRecognition] API unavailable")
                break

threading.Thread(target=listen_for_speech, daemon=True).start()

# Initialize MediaPipe drawing and pose estimation modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle_rad = math.acos(dot / (mag1 * mag2))
    return math.degrees(angle_rad)

# Smoothing function to reduce jitter
def smooth(prev, current, alpha=0.2):
    return (1 - alpha) * prev + alpha * current

def normalize_lighting(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv.merge((cl, a, b))
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)

def percent_change(a, b):
    return 100 * (a-b) / b

# Store manually editable shoulder points (None means use MediaPipe values)
manual_left_shoulder = None
manual_right_shoulder = None
editing_shoulder = None  # Track which shoulder is being edited ("left" or "right")

# Mouse callback function to allow clicking and dragging shoulder points
def mouse_callback(event, x, y, flags, param):
    global manual_left_shoulder, manual_right_shoulder, editing_shoulder

    # If mouse is pressed down, check if it is near either shoulder
    if event == cv.EVENT_LBUTTONDOWN:
        # Distance from left shoulder
        if manual_left_shoulder and abs(manual_left_shoulder[0] - x) < 20 and abs(manual_left_shoulder[1] - y) < 20:
            editing_shoulder = "left"
        elif manual_right_shoulder and abs(manual_right_shoulder[0] - x) < 20 and abs(manual_right_shoulder[1] - y) < 20:
            editing_shoulder = "right"

    # If mouse is being dragged, update the corresponding shoulder position
    elif event == cv.EVENT_MOUSEMOVE and editing_shoulder:
        if editing_shoulder == "left":
            manual_left_shoulder = (x, y)
        elif editing_shoulder == "right":
            manual_right_shoulder = (x, y)

    # If mouse is released, stop editing
    elif event == cv.EVENT_LBUTTONUP:
        editing_shoulder = None

# Create video capture object for webcam
cap = cv.VideoCapture(0)

# Set up mouse callback window to enable interactive shoulder alignment
cv.namedWindow('Posture Detection')
cv.setMouseCallback('Posture Detection', mouse_callback)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    mode = "front"
    prev_slouch_angle = 0
    prev_z_diff_nose = 0
    prev_spine_angle = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue
        frame = normalize_lighting(frame)

        posture_status = "No pose detected"
        color = (128, 128, 128)

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Check if landmarks were detected
        if results.pose_landmarks and results.pose_world_landmarks:
            # Use 2D landmarks (normalized coordinates)
            landmarks = results.pose_landmarks.landmark
            # Use world landmarks (3D coordinates in meters for more accurate depth and angle)
            world_landmarks = results.pose_world_landmarks.landmark
            # Get shoulder positions from MediaPipe or use manually overridden ones
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

            mid_eye_z = (left_eye.z + right_eye.z) / 2
            z_diff_head_nose = nose.z - left_hip.z
            z_diff_head_eye = mid_eye_z - left_hip.z

            adjusted_left_shoulder_y = (left_shoulder.y + left_ear.y) / 2
            h, w, _ = image.shape  # ✅ Ensure w and h are defined BEFORE use

            if manual_left_shoulder:
                # Override with manually dragged position if set
                adjusted_left_shoulder = [manual_left_shoulder[0] / w, manual_left_shoulder[1] / h]
            else:
                adjusted_left_shoulder = [left_shoulder.x, adjusted_left_shoulder_y]

            torso_vector = [adjusted_left_shoulder[0] - left_hip.x,
                            adjusted_left_shoulder[1] - left_hip.y]
            vertical_vector = [0, -1]

            slouch_angle = calculate_angle(torso_vector, vertical_vector)
            z_diff = left_shoulder.z - left_hip.z
            slouch_angle = smooth(prev_slouch_angle, slouch_angle)
            z_diff_head_nose = smooth(prev_z_diff_nose, z_diff_head_nose)
            prev_slouch_angle = slouch_angle
            prev_z_diff_nose = z_diff_head_nose

            clavicle_length = np.linalg.norm(np.array((left_shoulder.x, left_shoulder.y)) - np.array((right_shoulder.x, right_shoulder.y)))

            mid_back = [(left_shoulder.x + right_hip.x) / 2,
                        (left_shoulder.y + right_hip.y) / 2]
            spine_vector = [mid_back[0] - left_hip.x, mid_back[1] - left_hip.y]
            # Recalculate spine angle using real-world coordinates for accuracy
            world_left_shoulder = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            world_right_hip = world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            world_left_hip = world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            # Create a mid-back point in 3D space between left shoulder and right hip
            mid_back_world = [
                (world_left_shoulder.x + world_right_hip.x) / 2,
                (world_left_shoulder.y + world_right_hip.y) / 2
            ]
            spine_vector = [mid_back_world[0] - world_left_hip.x, mid_back_world[1] - world_left_hip.y]
            spine_angle = calculate_angle(spine_vector, vertical_vector)
            spine_angle = smooth(prev_spine_angle, spine_angle)
            # Measure sitting height using distance from nose to midpoint between hips
            mid_hip_y = (left_hip.y + right_hip.y) / 2
            sitting_height = abs(nose.y - mid_hip_y)
            head_to_shoulder_height = abs(nose.y - adjusted_left_shoulder_y)
            prev_spine_angle = spine_angle

            # Get frame dimensions for scaling
            h, w, _ = image.shape

            # If manual points exist, draw them visibly on screen
            if manual_left_shoulder:
                cv.circle(image, manual_left_shoulder, 8, (0, 255, 255), -1)
            if manual_right_shoulder:
                cv.circle(image, manual_right_shoulder, 8, (255, 255, 0), -1)
            cv.line(
                image,
                (int(mid_back[0] * w), int(mid_back[1] * h)),
                (int(left_hip.x * w), int(left_hip.y * h)),
                (0, 255, 0), 2
            )

            if is_calibrating:
                names = locals()
                elapsed = time.time() - calibration_start_time
                if elapsed < countdown_duration:
                    remaining = int(countdown_duration - elapsed) + 1
                    cv.putText(image, f"Starting in: {remaining}s", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif elapsed < countdown_duration + hold_duration:

                    for k in calibration_targets:
                        if k[:-1] in names:
                            calibration_data[k].append(names[k[:-1]])
                        else:
                            print(f"Calibration target {k[:-1]} not found in environment. Could not calibrate")
                    cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    is_calibrating = False
                    cv.putText(image, "Calibration complete!", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print("Calibration ended. Processing...")
                    if len(calibration_data["slouch_angles"]) > 0:
                        for k in calibration_targets:
                            calibrated_avgs[k] = sum(calibration_data[k]) / len(calibration_data[k])
                        print("Calibration complete.")
                        print("Averages:", calibrated_avgs)

            if calibrated_avgs:
                mode = "side" if (clavicle_length / calibrated_avgs["clavicle_lengths"]) < 0.6 else "front"
                if mode == "front":
                    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
                    # Diagnostic messages for specific posture deviations
                    if shoulder_tilt > 0.2:
                        posture_status = "Shoulder Tilt / Crunch Detected"
                        color = (0, 0, 255)
                    elif percent_change(head_to_shoulder_height, calibrated_avgs["head_to_shoulder_heights"]) > 10.0:
                        change = percent_change(head_to_shoulder_height, calibrated_avgs["head_to_shoulder_heights"])
                        print(f"Tilt detected {change}")
                        posture_status = "Head Tilt Detected"
                        color = (0, 0, 255)
                    elif percent_change(sitting_height, calibrated_avgs["sitting_heights"]) > 10.0:
                        change = percent_change(sitting_height, calibrated_avgs["sitting_heights"])
                        print(f"Sitting change detected {change}")
                        posture_status = "Overall Height Decrease (Possible Slouch)"
                        color = (0, 0, 255)
                    elif percent_change(slouch_angle, calibrated_avgs["slouch_angles"]) > 5.0:
                        change = percent_change(slouch_angle, calibrated_avgs["slouch_angles"])
                        print(f"Slouch detected {change}")
                        posture_status = "Slouching!"
                        color = (0, 0, 255)
                    else:
                        posture_status = "Great Posture!"
                        color = (0, 255, 0)
                elif mode == "side":
                    # STRAIGHTNESS AND FORWARD TILT CHECK: Detect spinal curve and shoulder tilt
                    if percent_change(spine_angle, calibrated_avgs["spine_angles"]) > 5.0 or slouch_angle > 15:
                        posture_status = "Hunched Spine or Forward Tilt"
                        color = (0, 0, 255)
                    elif percent_change(slouch_angle, calibrated_avgs["spine_angles"]) > 2.5 or slouch_angle > 10:
                        posture_status = "Moderate Curve or Lean"
                        color = (0, 165, 255)
                    else:
                        posture_status = "Good Side Posture"
                        color = (0, 255, 0)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, posture_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Display live spine angle on screen for visual feedback
            cv.putText(image, f"Spine Angle: {round(spine_angle, 1)} deg", (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        cv.imshow('Posture Detection', image)

        key = cv.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            mode = "side" if mode == "front" else "front"
            print("Switched to", mode)
        elif key == ord('c'):
            calibration_start_time = time.time()
            countdown_duration = 3
            hold_duration = 5
            is_calibrating = True
            calibration_data = {k: [] for k in calibration_data}
            print("Calibration countdown started. Get ready...")

cap.release()
cv.destroyAllWindows()