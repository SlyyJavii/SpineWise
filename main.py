import cv2 as cv
import time
import csv 
from datetime import datetime
import mediapipe as mp
import math
import speech_recognition as sr
import threading

last_log_time = time.time()
log_file = open("posture_trend_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

is_calibrating = False
calibration_data = {
    "slouch_angles": [],
    "z_diffs": [],
    "nose_hip_z_diffs": [],
    "eye_hip_z_diffs": [],
    "spine_angles": [],
    "sitting_heights": [],
    "head_to_shoulder_heights": [],
    "shoulder_mouth_distances": []
}
countdown_duration = 3
hold_duration = 5
calibrated_thresholds = {}

mode = "front"

def listen_for_speech():
    global is_calibrating, calibration_data, calibration_start_time, mode, countdown_duration, hold_duration
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
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
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
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


        confidence_score = 0  # already declared inside 'front' and 'side'
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
            head_tilt_difference = left_ear.y - right_ear.y
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
            mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
            mouth = type(mouth_left)(x=(mouth_left.x + mouth_right.x) / 2,
                         y=(mouth_left.y + mouth_right.y) / 2,
                         z=(mouth_left.z + mouth_right.z) / 2,
                         visibility=1.0)



            mid_eye_z = (left_eye.z + right_eye.z) / 2
            z_diff_head_nose = nose.z - left_hip.z
            z_diff_head_eye = mid_eye_z - left_hip.z
            adjusted_left_shoulder_y = (left_shoulder.y + left_ear.y) / 2

            torso_vector = [left_shoulder.x - left_hip.x, adjusted_left_shoulder_y - left_hip.y]
            vertical_vector = [0, -1]

            slouch_angle = calculate_angle(torso_vector, vertical_vector)
            z_diff = left_shoulder.z - left_hip.z
            slouch_angle = smooth(prev_slouch_angle, slouch_angle)
            z_diff_head_nose = smooth(prev_z_diff_nose, z_diff_head_nose)
            prev_slouch_angle = slouch_angle
            prev_z_diff_nose = z_diff_head_nose

            mid_back = [(left_shoulder.x + right_hip.x) / 2, (left_shoulder.y + right_hip.y) / 2]
            world_left_shoulder = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            world_right_hip = world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            world_left_hip = world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            mid_back_world = [(world_left_shoulder.x + world_right_hip.x) / 2, (world_left_shoulder.y + world_right_hip.y) / 2]
            spine_vector = [mid_back_world[0] - world_left_hip.x, mid_back_world[1] - world_left_hip.y]
            spine_angle = calculate_angle(spine_vector, vertical_vector)
            spine_angle = smooth(prev_spine_angle, spine_angle)
            prev_spine_angle = spine_angle

            mid_hip_y = (left_hip.y + right_hip.y) / 2
            sitting_height = abs(nose.y - mid_hip_y)
            head_to_shoulder_height = abs(nose.y - adjusted_left_shoulder_y)

            if is_calibrating:
                elapsed = time.time() - calibration_start_time
                if elapsed < countdown_duration:
                    remaining = int(countdown_duration - elapsed) + 1
                    cv.putText(image, f"Starting in: {remaining}s", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif elapsed < countdown_duration + hold_duration:
                    left_shoulder_z = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                    right_shoulder_z = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                    avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
                    calibration_data.setdefault("shoulder_z_depths", []).append(avg_shoulder_z)
                    calibration_data["slouch_angles"].append(slouch_angle)
                    calibration_data["z_diffs"].append(z_diff)
                    calibration_data["nose_hip_z_diffs"].append(z_diff_head_nose)
                    calibration_data["eye_hip_z_diffs"].append(z_diff_head_eye)
                    calibration_data["spine_angles"].append(spine_angle)
                    calibration_data["sitting_heights"].append(sitting_height)
                    calibration_data["head_to_shoulder_heights"].append(head_to_shoulder_height)
                    left_shoulder_mouth_dist = abs(left_shoulder.y - mouth.y)
                    right_shoulder_mouth_dist = abs(right_shoulder.y - mouth.y)
                    avg_shoulder_mouth_dist = (left_shoulder_mouth_dist + right_shoulder_mouth_dist) / 2
                    left_shoulder_z = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                    right_shoulder_z = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                    avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2

                    calibration_data.setdefault("shoulder_mouth_distances", []).append(avg_shoulder_mouth_dist)

                    cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    is_calibrating = False
                    avg = lambda k: sum(calibration_data[k]) / len(calibration_data[k]) if calibration_data[k] else 0
                    calibrated_thresholds = {
                        "slouch_warn": avg("slouch_angles") + 10,
                        "slouch_bad": avg("slouch_angles") + 20,
                        "z_warn": avg("z_diffs") - 0.05,
                        "z_bad": avg("z_diffs") - 0.15,
                        "nose_warn": avg("nose_hip_z_diffs") - 0.1,
                        "nose_bad": avg("nose_hip_z_diffs") - 0.25,
                        "spine_bad": avg("spine_angles") + 15,
                        "height_drop_threshold": avg("sitting_heights") - 0.02,
                        "head_shoulder_drop_threshold": avg("head_to_shoulder_heights") - 0.015,
                        "shoulder_mouth_warn": avg("shoulder_mouth_distances") - 0.015,
                        "shoulder_z_lean_threshold": avg("shoulder_z_depths") - 0.05


                        
                    }
                    nose_z = world_landmarks[mp_pose.PoseLandmark.NOSE].z
                    hip_z = (
                                world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].z +
                                     world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z
                        ) / 2
                    z_diff_nose_hip = nose_z - hip_z

            if calibrated_thresholds:
                if mode == "front":



                    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
                    shoulder_tilt_threshold = 0.015 + calibrated_thresholds["slouch_warn"] / 200

                    confidence_score = 0
                    head_tilt_difference = left_ear.y - right_ear.y
                    head_tilt_status = None

                    if head_tilt_difference > 0.025:
                        head_tilt_status = "Head Tilted Right"
                        confidence_score += 1
                    elif head_tilt_difference < -0.025:
                        head_tilt_status = "Head Tilted Left"
                        confidence_score += 1

                    left_shoulder_mouth_dist = abs(left_shoulder.y - mouth.y)
                    right_shoulder_mouth_dist = abs(right_shoulder.y - mouth.y)
                    avg_shoulder_mouth_dist = (left_shoulder_mouth_dist + right_shoulder_mouth_dist) / 2

                    nose_z = world_landmarks[mp_pose.PoseLandmark.NOSE].z
                    hip_z = (
                                world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].z +
                                world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z
                             ) / 2
                    z_diff_nose_hip = nose_z - hip_z


                    if z_diff_nose_hip < calibrated_thresholds["nose_bad"] + 0.02:
                        confidence_score += 1 #forward lead
                    if slouch_angle > calibrated_thresholds["slouch_warn"] - 5:
                        confidence_score += 1
                    if spine_angle > calibrated_thresholds["spine_bad"] - 5:
                        confidence_score += 1
                    if shoulder_tilt > shoulder_tilt_threshold:
                        confidence_score += 1
                    if head_to_shoulder_height < calibrated_thresholds["head_shoulder_drop_threshold"] + 0.01:
                        confidence_score += 1
                    if avg_shoulder_mouth_dist < calibrated_thresholds["shoulder_mouth_warn"]:
                        confidence_score += 1
                    left_shoulder_z = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                    right_shoulder_z = world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                    avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2

                    if avg_shoulder_z < calibrated_thresholds["shoulder_z_lean_threshold"]:
                        confidence_score += 1
                        shoulder_leaning_forward = True
                    else:
                        shoulder_leaning_forward = False



                    
                    if confidence_score >= 5:
                        new_status = "Critical Forward Posture"
                        new_color = (128,0,0) #dark red
                    elif confidence_score == 4:
                        new_status = "Severe Slouch + Lean"
                        new_color = (0, 0, 255)
                    elif confidence_score == 3:
                        new_status = "Significant Postural Issue"
                        new_color = (0, 165, 255)
                    elif confidence_score == 2:
                        new_status = "Early posture warning"
                        new_color = (255, 165, 0)
                    elif z_diff_nose_hip < calibrated_thresholds["nose_bad"] + 0.02:
                        new_status = "Leaning forward"
                        new_color = (0, 140, 255)
                    elif avg_shoulder_mouth_dist < calibrated_thresholds["shoulder_mouth_warn"] - 0.01:
                        new_status = "Shoulder Creep Detected"
                        new_color = (255, 105, 180)  # pink
                    elif head_tilt_status == "Head Tilted Left":
                        new_status = "Head Tilted Left"
                        new_color = (255, 215, 0)
                    elif head_tilt_status == "Head Tilted Right":
                        new_status = "Head Tilted Right"
                        new_color = (255, 165, 0)
                    elif shoulder_leaning_forward:
                        new_status = "Shoulders Too Close to Screen"
                        new_color = (0, 100, 255)



                    else:
                        new_status = "Good Posture"
                        new_color = (0, 255, 0)

            elif mode == "side":
                left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

                # Determine which shoulder is closer to screen center
                facing = "left" if left_shoulder_x > right_shoulder_x else "right"
                if facing == "left":
                    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    side_label = "Right Side View"
                else:
                    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    side_label = "Left Side View"

                # Calculate a side-specific slouch angle
                slouch_vector = [shoulder.x - hip.x, shoulder.y - hip.y]
                slouch_angle = calculate_angle(slouch_vector, [0, -1])




                if spine_angle > calibrated_thresholds["spine_bad"] + 15:
                    new_status = "Critical Spinal Collapse"
                    new_color = (139, 0, 0)  # dark red
                elif spine_angle > calibrated_thresholds["spine_bad"] + 10:
                    new_status = "Severe Spinal Hunch"
                    new_color = (255, 0, 0)
                elif spine_angle > calibrated_thresholds["spine_bad"]:
                    new_status = "Hunched Spine or Forward Tilt"
                    new_color = (255, 69, 0)  # orange red
                elif spine_angle > calibrated_thresholds["spine_bad"] - 10:
                    new_status = "Moderate Curve or Lean"
                    new_color = (255, 140, 0)
                elif spine_angle > calibrated_thresholds["spine_bad"] - 15:
                   new_status = "Mild Curve"
                   new_color = (255, 215, 0)
                elif slouch_angle > calibrated_thresholds["slouch_bad"]:
                    new_status = "Forward Head Posture"
                    new_color = (0, 191, 255)
                elif slouch_angle > calibrated_thresholds["slouch_warn"]:
                    new_status = "Slight Forward Lean"
                    new_color = (135, 206, 250)
                else:
                    if confidence_score >= 4:
                        new_status = "Critical Spinal Collapse"
                        new_color = (139, 0, 0)
                    elif confidence_score == 3:
                        new_status = "Severe Spinal Hunch"
                        new_color = (255, 0, 0)
                    elif confidence_score == 2:
                        new_status = "Moderate Forward Curve"
                        new_color = (255, 140, 0)
                    elif confidence_score == 1:
                        new_status = "Minor Curve or Head Tilt"
                        new_color = (173, 216, 230)
                    else:
                        new_status = "Neutral Side Posture"
                        new_color = (0, 255, 127)

            if new_status != current_status:
                current_status = new_status
                last_status_change_time = time.time()
            elif time.time() - last_status_change_time > 1:
                displayed_status = current_status
                displayed_color = new_color

                if time.time() - last_log_time >= 10:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([
                                            timestamp,
                                            mode,
                                            facing if mode == "side" else "front",
                                            current_status,
                                            head_tilt_status if head_tilt_status else "neutral",
                                            confidence_score
                                        ])
                    log_file.flush()
                    last_log_time = time.time()


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
            cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, displayed_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, displayed_color, 2)
            if head_tilt_status:
                cv.putText(image, head_tilt_status, (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            if shoulder_leaning_forward:
                cv.putText(image, "Leaning In", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv.rectangle(image, (30, 180), (30 + confidence_score * 40, 200), (0, 255 - confidence_score * 50, 50), -1)
            cv.putText(image, f"Confidence: {confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)




        cv.imshow('Posture Detection', image)
        key = cv.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            mode = "side" if mode == "front" else "front"
        elif key == ord('c'):
            calibration_start_time = time.time()
            is_calibrating = True
            calibration_data = {k: [] for k in calibration_data}

cap.release()
cv.destroyAllWindows()
log_file.close()




