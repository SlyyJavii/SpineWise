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

pygame.init()
pygame.mixer.init()

beep = pygame.mixer.Sound("bad_posture_alert.wav")

#Globals for posture tracking 
start_time = None
loop_started = False 
last_beep_time = 0

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
    "clavicle_y": [],
    "face_mesh_tilt": [],      # Face Mesh head tilt baseline
    "face_mesh_rotation": [],  # Face Mesh head rotation baseline  
    "face_mesh_lean": [],      # Face Mesh forward lean baseline
    "face_mesh_distance": [],  # Face Mesh distance baseline
    "raw_eye_tilt_samples": [], # NEW: Store raw tilt samples to detect natural asymmetry
    "nose_eye_distances": []   # NEW: Store nose-to-eye baseline for looking down detection
}
countdown_duration = 3
hold_duration = 5
calibrated_thresholds = {}
calibration_start_time = 0
bad_posture_start_time = None
alert_active = False

mode = "front"
prev_facial_distances = [0, 0, 0]  # Reduced cache size for faster response
prev_torso_distances = [0, 0, 0]   # Reduced cache size for faster response
cache_idx = 0

# Posture stability system
posture_buffer_size = 15  # Number of frames to track
posture_buffer = []  # Store recent posture statuses
confidence_buffer = []  # Store recent confidence scores
current_stable_posture = "good"  # Current confirmed posture state
posture_transition_time = {}  # Track how long each posture has been candidate
min_transition_duration = 2.0  # Seconds before confirming posture change
smoothed_confidence = 0.0  # Exponentially smoothed confidence score
confidence_alpha = 0.3  # Smoothing factor (lower = more smoothing)
calibration_grace_period = 3.0  # Seconds after calibration before enabling transitions
calibration_end_time = None  # When calibration last ended

# Face Mesh landmark indices for head orientation
FACE_LANDMARKS = {
    "left_eye_corner": 33,    # Left eye outer corner
    "right_eye_corner": 263,  # Right eye outer corner
    "left_eye_center": 159,   # Left eye center
    "right_eye_center": 386,  # Right eye center
    "nose_tip": 1,            # Nose tip
    "nose_bridge": 6,         # Nose bridge
    "chin": 175,              # Chin center
    "forehead": 10,           # Forehead center
    "left_cheek": 234,        # Left cheek
    "right_cheek": 454,       # Right cheek
    "left_temple": 127,       # Left temple
    "right_temple": 356       # Right temple
}
posture_status_labels = {
    "good": ("Good Posture", (0, 255, 0)),
    "moderate": ("Moderately Bad Posture", (255, 165, 0)),  # Orange
    "bad": ("Bad Posture", (255, 0, 0))  # Red
}

def update_posture_stability(confidence_score, max_score=7):
    """
    Update posture stability system with consistency buffer and state confirmation.
    Returns: (stable_posture_status, display_status, is_transitioning)
    """
    global posture_buffer, confidence_buffer, current_stable_posture, posture_transition_time
    global smoothed_confidence, calibration_end_time
    
    current_time = time.time()
    
    # Check if we're in calibration grace period
    if calibration_end_time and (current_time - calibration_end_time) < calibration_grace_period:
        grace_remaining = calibration_grace_period - (current_time - calibration_end_time)
        return current_stable_posture, f"Stabilizing... ({grace_remaining:.1f}s)", True
    
    # Apply exponential smoothing to confidence score
    smoothed_confidence = confidence_alpha * confidence_score + (1 - confidence_alpha) * smoothed_confidence
    
    # Get current frame's posture status based on smoothed confidence
    current_frame_posture = get_posture_status(int(smoothed_confidence), max_score)
    
    # Add to buffers
    posture_buffer.append(current_frame_posture)
    confidence_buffer.append(smoothed_confidence)
    
    # Maintain buffer size
    if len(posture_buffer) > posture_buffer_size:
        posture_buffer.pop(0)
        confidence_buffer.pop(0)
    
    # If buffer isn't full yet, stay with current stable posture
    if len(posture_buffer) < posture_buffer_size:
        return current_stable_posture, posture_status_labels[current_stable_posture][0], False
    
    # Count occurrences of each posture in recent frames
    posture_counts = {"good": 0, "moderate": 0, "bad": 0}
    for posture in posture_buffer:
        posture_counts[posture] += 1
    
    # Find the most frequent posture in recent frames
    candidate_posture = max(posture_counts, key=posture_counts.get)
    candidate_confidence = posture_counts[candidate_posture] / len(posture_buffer)
    
    # Track transition timing
    if candidate_posture != current_stable_posture:
        # New candidate posture detected
        if candidate_posture not in posture_transition_time:
            posture_transition_time[candidate_posture] = current_time
        
        # Check if candidate has been consistent long enough
        transition_duration = current_time - posture_transition_time[candidate_posture]
        confidence_threshold = 0.6 if candidate_posture == "bad" else 0.7  # Slightly easier to detect bad posture
        
        if transition_duration >= min_transition_duration and candidate_confidence >= confidence_threshold:
            # Confirm the transition
            print(f"[POSTURE] Confirmed transition: {current_stable_posture} → {candidate_posture} "
                  f"(confidence: {candidate_confidence:.2f}, duration: {transition_duration:.1f}s)")
            current_stable_posture = candidate_posture
            # Clear transition timers
            posture_transition_time = {}
        else:
            # Still transitioning
            remaining_time = max(0, min_transition_duration - transition_duration)
            return current_stable_posture, f"Detecting {candidate_posture}... ({remaining_time:.1f}s)", True
    else:
        # Current posture is stable, clear any transition timers
        posture_transition_time = {}
    
    # Return stable posture
    status_text = posture_status_labels[current_stable_posture][0]
    return current_stable_posture, status_text, False

def get_posture_status(confidence_score, max_score=7):
    """Convert confidence score to simplified posture status"""
    if confidence_score <= 1:
        return "good"
    elif confidence_score <= 3:
        return "moderate"
    else:
        return "bad"

def calculate_head_metrics_from_face_mesh(face_landmarks):
    """
    Calculate head orientation metrics using Face Mesh landmarks.
    Returns: (head_tilt, head_rotation, head_forward_lean, face_center_z, tilt_direction, nose_eye_distance, left_cheek, right_cheek)
    """
    if not face_landmarks or len(face_landmarks) == 0:
        return 0, 0, 0, 0, "none", 0, None, None
    
    landmarks = face_landmarks[0]  # Get first face
    
    # Debug: Check landmark count
    if len(landmarks) < 468:
        print(f"[DEBUG] Insufficient face landmarks: {len(landmarks)}/468")
        return 0, 0, 0, 0, "none", 0, None, None
    
    try:
        # Extract key landmarks
        left_eye = landmarks[FACE_LANDMARKS["left_eye_center"]]
        right_eye = landmarks[FACE_LANDMARKS["right_eye_center"]]
        nose_tip = landmarks[FACE_LANDMARKS["nose_tip"]]
        nose_bridge = landmarks[FACE_LANDMARKS["nose_bridge"]]
        chin = landmarks[FACE_LANDMARKS["chin"]]
        forehead = landmarks[FACE_LANDMARKS["forehead"]]
        left_temple = landmarks[FACE_LANDMARKS["left_temple"]]
        right_temple = landmarks[FACE_LANDMARKS["right_temple"]]
        left_cheek = landmarks[FACE_LANDMARKS["left_cheek"]]   # Use as ear replacement
        right_cheek = landmarks[FACE_LANDMARKS["right_cheek"]] # Use as ear replacement
        
        # 1. HEAD TILT: Calculate raw difference using CHEEKS instead of ears
        raw_eye_tilt = left_cheek.y - right_cheek.y  # Positive = tilted left, Negative = tilted right
        head_tilt_absolute = abs(raw_eye_tilt)   # Always positive for scoring
        
        # Determine tilt direction for debugging
        if abs(raw_eye_tilt) < 0.005:  # Very small difference, consider straight
            tilt_direction = "straight"
        elif raw_eye_tilt > 0:
            tilt_direction = "left"
        else:
            tilt_direction = "right"
        
        # 2. HEAD ROTATION: Temple depth difference (left/right turn)
        head_rotation = abs(left_temple.z - right_temple.z)
        
        # 3. FORWARD HEAD LEAN: Nose tip vs nose bridge depth
        # When leaning forward, nose tip comes closer to camera than bridge
        head_forward_lean = nose_bridge.z - nose_tip.z
        
        # 4. Overall face center Z for distance tracking
        face_center_z = (left_eye.z + right_eye.z + nose_tip.z + chin.z) / 4
        
        # 5. LOOKING DOWN DETECTION - Nose to eye center distance
        eye_center_y = (left_eye.y + right_eye.y) / 2
        nose_eye_distance = nose_tip.y - eye_center_y  # Positive when nose is below eyes (looking down)
        
        return head_tilt_absolute, head_rotation, head_forward_lean, face_center_z, tilt_direction, nose_eye_distance, left_cheek, right_cheek
        
    except (IndexError, AttributeError) as e:
        print(f"[DEBUG] Error accessing face landmarks: {e}")
        return 0, 0, 0, 0, "error", 0, None, None

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
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
        ])
        drawing_utils.draw_landmarks(
            frame,
            landmarks_proto,
            connections,
            landmark_style)

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

def analyze_posture(image, pose_landmarks, face_landmarks=None):
    global is_calibrating, calibration_data, calibrated_thresholds, calibration_start_time, mode, countdown_duration, hold_duration, prev_facial_distances, prev_torso_distances, cache_idx
    global start_time, loop_started, last_beep_time

    side_label = ""
    status = "No pose detected"
    color = (128, 128, 128)
    
    # Initialize confidence scores to prevent UnboundLocalError
    head_confidence_score = 0
    body_confidence_score = 0
    side_confidence_score = 0
    combined_confidence = 0

    left_shoulder = pose_landmarks[PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[PoseLandmark.RIGHT_SHOULDER]
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return
        
    left_hip = pose_landmarks[PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[PoseLandmark.RIGHT_HIP]
    left_ear = pose_landmarks[PoseLandmark.LEFT_EAR]
    right_ear = pose_landmarks[PoseLandmark.RIGHT_EAR]

    nose = pose_landmarks[PoseLandmark.NOSE]
    left_eye = pose_landmarks[PoseLandmark.LEFT_EYE]
    right_eye = pose_landmarks[PoseLandmark.RIGHT_EYE]
    mouth_left = pose_landmarks[PoseLandmark.MOUTH_LEFT]
    mouth_right = pose_landmarks[PoseLandmark.MOUTH_RIGHT]
    
    # Calculate averaged positions
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

    # Update rolling averages (reduced cache for faster response)
    prev_torso_distances[cache_idx] = torso.z
    prev_facial_distances[cache_idx] = face.z
    cache_idx = (cache_idx + 1) % 3  # Reduced from 6 to 3 for faster response

    torso_distance = np.mean(prev_torso_distances).astype(float)
    facial_distance = np.mean(prev_facial_distances).astype(float)
    
    # Get Face Mesh head metrics (ENHANCED with looking down detection and cheek landmarks)
    face_tilt, face_rotation, face_lean, face_distance, tilt_direction, nose_eye_distance, left_cheek, right_cheek = calculate_head_metrics_from_face_mesh(face_landmarks)
    
    # NEW: Use Face Mesh cheeks as ear replacements for better accuracy
    if left_cheek is not None and right_cheek is not None:
        # Use Face Mesh cheek landmarks instead of pose ear landmarks
        left_ear = left_cheek
        right_ear = right_cheek
        raw_head_tilt = left_cheek.y - right_cheek.y  # Face Mesh based tilt
        print(f"[DEBUG] Using Face Mesh cheeks for ear positions")
    else:
        # Fallback to pose-based ears if Face Mesh fails
        raw_head_tilt = left_ear.y - right_ear.y if left_ear.visibility > 0.5 and right_ear.visibility > 0.5 else 0
        print(f"[DEBUG] Fallback to pose ear landmarks")
    
    head_tilt_difference = abs(raw_head_tilt)

    clavicle_length = np.linalg.norm(
        np.array((left_shoulder.x, left_shoulder.y)) - np.array((right_shoulder.x, right_shoulder.y)))
    
    # Calculate shoulder-ear distances using Face Mesh cheeks when available
    if left_cheek is not None and right_cheek is not None:
        # Use Face Mesh cheek landmarks for more accurate shoulder-ear calculations
        left_shoulder_ear = np.linalg.norm(np.array((left_shoulder.x, left_shoulder.y)) - np.array((left_cheek.x, left_cheek.y)))
        right_shoulder_ear = np.linalg.norm(np.array((right_shoulder.x, right_shoulder.y)) - np.array((right_cheek.x, right_cheek.y)))
    else:
        # Fallback to pose ear landmarks
        left_shoulder_ear = np.linalg.norm(np.array((left_shoulder.x, left_shoulder.y)) - np.array((left_ear.x, left_ear.y)))
        right_shoulder_ear = np.linalg.norm(np.array((right_shoulder.x, right_shoulder.y)) - np.array((right_ear.x, right_ear.y)))
    
    avg_shoulder_ear = (left_shoulder_ear + right_shoulder_ear) / 2

    face_torso_height = face.y - torso.y

    avg = lambda k: sum(calibration_data[k]) / len(calibration_data[k]) if calibration_data[k] else 0

    # Calibration logic
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
            
            # Calibrate Face Mesh baselines
            calibration_data["face_mesh_tilt"].append(face_tilt)
            calibration_data["face_mesh_rotation"].append(face_rotation)
            calibration_data["face_mesh_lean"].append(face_lean)
            calibration_data["face_mesh_distance"].append(face_distance)
            
            # NEW: Calibrate nose-eye distance baseline for looking down detection
            calibration_data["nose_eye_distances"].append(nose_eye_distance)
            
            # Store raw tilt samples using Face Mesh cheeks when available
            if face_landmarks and len(face_landmarks) > 0:
                landmarks = face_landmarks[0]
                if len(landmarks) >= 468:
                    # Use Face Mesh cheek landmarks for tilt calibration
                    left_cheek_landmark = landmarks[FACE_LANDMARKS["left_cheek"]]
                    right_cheek_landmark = landmarks[FACE_LANDMARKS["right_cheek"]]
                    raw_tilt_sample = left_cheek_landmark.y - right_cheek_landmark.y
                    calibration_data["raw_eye_tilt_samples"].append(raw_tilt_sample)
                else:
                    # Fallback to eye landmarks if cheeks not available
                    left_eye = landmarks[FACE_LANDMARKS["left_eye_center"]]
                    right_eye = landmarks[FACE_LANDMARKS["right_eye_center"]]
                    raw_tilt_sample = left_eye.y - right_eye.y
                    calibration_data["raw_eye_tilt_samples"].append(raw_tilt_sample)
            
            cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
            cv.putText(image, "Keep head straight and level!", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 1)
        else:
            is_calibrating = False
            
            # Calculate natural tilt baseline from calibration samples
            natural_tilt_baseline = 0
            if calibration_data["raw_eye_tilt_samples"]:
                natural_tilt_baseline = sum(calibration_data["raw_eye_tilt_samples"]) / len(calibration_data["raw_eye_tilt_samples"])
                tilt_variance = np.var(calibration_data["raw_eye_tilt_samples"])
                print(f"[CALIBRATION] Natural head tilt baseline: {natural_tilt_baseline:.4f}")
                print(f"[CALIBRATION] Tilt variance during calibration: {tilt_variance:.6f}")
            
            calibrated_thresholds = {
                "clavicle_length_threshold": avg("clavicle_lengths") * 0.7,
                # Face Mesh baselines
                "face_mesh_tilt_baseline": avg("face_mesh_tilt"),
                "face_mesh_rotation_baseline": avg("face_mesh_rotation"), 
                "face_mesh_lean_baseline": avg("face_mesh_lean"),
                "face_mesh_distance_baseline": avg("face_mesh_distance"),
                # Natural tilt baseline for symmetric detection
                "natural_tilt_baseline": natural_tilt_baseline,
                # NEW: Nose-eye distance baseline for looking down detection
                "nose_eye_baseline": avg("nose_eye_distances")
            }
            # Set calibration end time for grace period
            global calibration_end_time
            calibration_end_time = time.time()
            print(f"[POSTURE] Calibration complete. Grace period: {calibration_grace_period}s")

    # Main posture analysis
    if calibrated_thresholds and not is_calibrating:
        mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"

        if mode == "front":
            # Calculate baseline percentages
            facial_avg = avg("facial_distances")
            torso_avg = avg("torso_distances")
            face_clav_height_avg = avg("face_torso_heights")
            shoulder_ear_avg = avg("shoulder_ear_distance")
            clavicle_y_baseline = avg("clavicle_y")
            nose_eye_avg = calibrated_thresholds["nose_eye_baseline"]

            facial_percentage = (facial_distance - facial_avg) / facial_avg if facial_avg != 0 else 0
            torso_percentage = (torso_distance - torso_avg) / torso_avg if torso_avg != 0 else 0
            height_percentage = (face_torso_height - face_clav_height_avg) / face_clav_height_avg if face_clav_height_avg != 0 else 0
            shoulder_ear_percentage = (avg_shoulder_ear - shoulder_ear_avg) / shoulder_ear_avg if shoulder_ear_avg != 0 else 0
            
            # NEW: Looking down percentage using same pattern as other metrics
            looking_down_percentage = (nose_eye_distance - nose_eye_avg) / nose_eye_avg if nose_eye_avg != 0 else 0
            
            # ENHANCED: More responsive clavicle Y-drop detection
            clavicle_y_pct = (clavicle.y - clavicle_y_baseline) / clavicle_y_baseline if clavicle_y_baseline != 0 else 0
            # Also check absolute drop to catch cases where baseline calibration was poor
            absolute_clavicle_drop = clavicle.y - clavicle_y_baseline

            # FIXED: Proper symmetric head tilt calculation with baseline normalization
            head_tilt_baseline = calibrated_thresholds.get("head_tilt_baseline", 0)
            normalized_head_tilt = abs(raw_head_tilt - head_tilt_baseline)  # Remove baseline bias and use absolute value
            
            # Head-specific metrics (only head position and tilt)
            head_forward_score = scale_metric(facial_percentage, 0.08, 0.25, 3)  # Slightly more sensitive
            head_tilt_score = scale_metric(normalized_head_tilt, 0.02, 0.08, 3)   # More sensitive with proper normalization
            
            # NEW: Looking down score using same scale_metric function
            looking_down_score = scale_metric(looking_down_percentage, 0.40, 0.60, 4)  # 15%-40% change triggers penalty
            
            head_confidence_score = min(head_forward_score + head_tilt_score + looking_down_score, 7)

            # Debug head tilt when significant (now using Face Mesh cheeks)
            if normalized_head_tilt > 0.03:
                tilt_direction_str = "LEFT" if raw_head_tilt > head_tilt_baseline else "RIGHT"
                landmark_source = "Face Mesh cheeks" if (left_cheek is not None and right_cheek is not None) else "Pose ears"
                print(f"[HEAD TILT DEBUG] Raw: {raw_head_tilt:.4f}, Baseline: {head_tilt_baseline:.4f}, "
                      f"Normalized: {normalized_head_tilt:.4f}, Direction: {tilt_direction_str}, "
                      f"Source: {landmark_source}, Score: {head_tilt_score}/3")

            # Body-specific metrics with ENHANCED clavicle Y-drop sensitivity
            torso_lean_score = scale_metric(torso_percentage, 0.08, 0.25, 2)      
            shoulder_drop_score = scale_metric(shoulder_ear_percentage, 0.08, 0.20, 2)  
            
            # ENHANCED: Much more sensitive and weighted clavicle drop detection
            posture_drop_score_pct = scale_metric(clavicle_y_pct, 0.005, 0.03, 4)  # Very sensitive: 0.5%-3%
            posture_drop_score_abs = scale_metric(absolute_clavicle_drop, 0.01, 0.05, 4)  # Absolute threshold
            # Take the maximum of both methods for best detection
            posture_drop_score = min(max(posture_drop_score_pct, posture_drop_score_abs), 4)
            
            body_alignment_score = scale_metric(height_percentage, 0.08, 0.20, 1) 
            
            # ENHANCED: Give clavicle drop much more weight in final calculation
            body_confidence_score = min(torso_lean_score + shoulder_drop_score + body_alignment_score + posture_drop_score, 7)

            # Combine scores with clear weighting, but boost clavicle impact
            base_combined = int((head_confidence_score * 0.5 + body_confidence_score * 0.5))
            # Add extra penalty for significant posture drop
            clavicle_penalty = min(posture_drop_score // 2, 2)  # 0-2 extra points for severe drops
            combined_confidence = min(base_combined + clavicle_penalty, 7)

            # Debug output for clavicle tracking
            if posture_drop_score > 1:
                print(f"[CLAVICLE DEBUG] Y-drop: {clavicle_y_pct:.3f} ({posture_drop_score_pct}/4), "
                      f"Abs drop: {absolute_clavicle_drop:.3f} ({posture_drop_score_abs}/4), "
                      f"Final drop score: {posture_drop_score}/4, Penalty: +{clavicle_penalty}")

            # Get simplified status using stability system
            stable_posture, display_status, is_transitioning = update_posture_stability(combined_confidence, 7)
            
            # Use stable posture for alert logic (prevents false alarms)
            stable_confidence = 1 if stable_posture == "good" else (3 if stable_posture == "moderate" else 5)
            
            # Alert logic (using stable confidence to prevent false alarms)
            alert_threshold = 3
            alert_duration = 10
            beep_interval = 2

            if (stable_confidence >= alert_threshold or head_confidence_score >= alert_threshold) and not is_transitioning:
                if start_time is None:
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    if elapsed >= alert_duration:
                        if not loop_started:
                            print("[Alert] Bad posture detected for 10 seconds. Starting beep loop.")
                            loop_started = True

                if loop_started and time.time() - last_beep_time >= beep_interval:
                    beep.play()
                    last_beep_time = time.time()
            else:
                if loop_started:
                    beep.stop()
                    print("[Info] Posture corrected. Stopping beeps.")
                start_time = None
                loop_started = False

            # Set display values
            status = display_status
            color = posture_status_labels[stable_posture][1]

        elif mode == "side":
            # Side view analysis - using Face Mesh cheeks when available for ear calculations
            facing = "left" if left_shoulder.z > right_shoulder.z else "right"
            if facing == "left":
                shoulder = right_shoulder
                hip = right_hip
                # Use Face Mesh right cheek if available, otherwise fall back to pose ear
                ear = right_cheek if right_cheek is not None else right_ear
                side_label = "Right Side View (Face Mesh)" if right_cheek is not None else "Right Side View (Pose)"
            else:
                shoulder = left_shoulder
                hip = left_hip
                # Use Face Mesh left cheek if available, otherwise fall back to pose ear
                ear = left_cheek if left_cheek is not None else left_ear
                side_label = "Left Side View (Face Mesh)" if left_cheek is not None else "Left Side View (Pose)"

            shoulder_ear_avg = avg("shoulder_ear_distance")
            shoulder_ear_percentage = (avg_shoulder_ear - shoulder_ear_avg) / shoulder_ear_avg if shoulder_ear_avg != 0 else 0
            
            # Side-specific confidence scoring
            forward_head_score = scale_metric(shoulder_ear_percentage, 0.08, 0.25, 7)  # Use full 0-7 range
            side_confidence_score = forward_head_score

            # Get simplified status using stability system  
            stable_posture, display_status, is_transitioning = update_posture_stability(side_confidence_score, 7)
            
            # Set display values
            status = display_status
            color = posture_status_labels[stable_posture][1]

    # DEBUG: Visual face mesh overlay (optional - remove in production)
    debug_face = getattr(globals().get('builtins', {}), 'debug_face', False)  # Get global debug state
    if debug_face and face_landmarks and len(face_landmarks) > 0:
        print(f"[DEBUG] Drawing Face Mesh overlay, landmarks count: {len(face_landmarks[0])}")
        # Draw key Face Mesh landmarks
        h, w = image.shape[:2]
        landmarks = face_landmarks[0]
        
        try:
            # Draw eye centers
            left_eye = landmarks[FACE_LANDMARKS["left_eye_center"]]
            right_eye = landmarks[FACE_LANDMARKS["right_eye_center"]]
            cv.circle(image, (int(left_eye.x * w), int(left_eye.y * h)), 5, (255, 0, 0), -1)  # Blue left eye
            cv.circle(image, (int(right_eye.x * w), int(right_eye.y * h)), 5, (0, 255, 0), -1)  # Green right eye
            
            # Draw nose tip and bridge
            nose_tip = landmarks[FACE_LANDMARKS["nose_tip"]]
            nose_bridge = landmarks[FACE_LANDMARKS["nose_bridge"]]
            cv.circle(image, (int(nose_tip.x * w), int(nose_tip.y * h)), 5, (0, 0, 255), -1)    # Red nose tip
            cv.circle(image, (int(nose_bridge.x * w), int(nose_bridge.y * h)), 5, (255, 255, 0), -1)  # Yellow bridge
            
            # Draw cheek landmarks (now used as ear replacements)
            left_cheek_landmark = landmarks[FACE_LANDMARKS["left_cheek"]]
            right_cheek_landmark = landmarks[FACE_LANDMARKS["right_cheek"]]
            cv.circle(image, (int(left_cheek_landmark.x * w), int(left_cheek_landmark.y * h)), 4, (0, 255, 255), -1)  # Cyan cheeks
            cv.circle(image, (int(right_cheek_landmark.x * w), int(right_cheek_landmark.y * h)), 4, (0, 255, 255), -1)
            
            # Show Face Mesh metrics
            cv.putText(image, f"Face Tilt: {face_tilt:.4f}", (w//2 - 80, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(image, f"Face Rotation: {face_rotation:.4f}", (w//2 - 80, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(image, f"Face Lean: {face_lean:.4f}", (w//2 - 80, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(image, f"Looking Down: {nose_eye_distance:.4f}", (w//2 - 80, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(image, f"Head Tilt (Cheeks): {raw_head_tilt:.4f}", (w//2 - 80, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"[DEBUG] Face metrics - Tilt: {face_tilt:.4f}, Rotation: {face_rotation:.4f}, Lean: {face_lean:.4f}, Looking Down: {nose_eye_distance:.4f}, Head Tilt (Cheeks): {raw_head_tilt:.4f}")
        except Exception as e:
            print(f"[DEBUG] Error drawing face mesh overlay: {e}")
    elif debug_face:
        print("[DEBUG] Face Mesh debug enabled but no landmarks available")

    # DEBUG: Visual cheek position overlay (replaces ear overlay since we're using cheeks as ears)
    debug_cheeks = False  # Set to True to see cheek landmark positions
    if debug_cheeks and calibrated_thresholds and left_cheek is not None and right_cheek is not None:
        # Draw horizontal lines at cheek positions (now used as ear replacements)
        h, w = image.shape[:2]
        left_cheek_pixel_y = int(left_cheek.y * h)
        right_cheek_pixel_y = int(right_cheek.y * h)
        
        cv.line(image, (0, left_cheek_pixel_y), (w//3, left_cheek_pixel_y), (255, 0, 0), 2)  # Blue line for left cheek
        cv.line(image, (2*w//3, right_cheek_pixel_y), (w, right_cheek_pixel_y), (0, 255, 0), 2)  # Green line for right cheek
        
        # Show tilt values
        cv.putText(image, f"LC: {left_cheek.y:.3f}", (10, left_cheek_pixel_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.putText(image, f"RC: {right_cheek.y:.3f}", (2*w//3 + 10, right_cheek_pixel_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(image, f"Cheek Tilt: {raw_head_tilt:.4f}", (w//2 - 60, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    elif debug_cheeks and calibrated_thresholds:
        # Fallback debug for pose ears if Face Mesh cheeks not available
        h, w = image.shape[:2]
        left_ear_pixel_y = int(left_ear.y * h)
        right_ear_pixel_y = int(right_ear.y * h)
        
        cv.line(image, (0, left_ear_pixel_y), (w//3, left_ear_pixel_y), (255, 0, 0), 2)  # Blue line for left ear
        cv.line(image, (2*w//3, right_ear_pixel_y), (w, right_ear_pixel_y), (0, 255, 0), 2)  # Green line for right ear
        
        # Show tilt values
        cv.putText(image, f"LE: {left_ear.y:.3f}", (10, left_ear_pixel_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.putText(image, f"RE: {right_ear.y:.3f}", (2*w//3 + 10, right_ear_pixel_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(image, f"Pose Tilt: {raw_head_tilt:.4f}", (w//2 - 60, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display information
    average_color = image[30:310, 175:220].mean((0, 1))
    final_color = ((255 - average_color[0]), (255 - average_color[1]), (255 - average_color[2]))

    cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (final_color[0], 220, final_color[2]), 2)

    # Only show confidence bars if calibration is complete
    if calibrated_thresholds and not is_calibrating:
        if mode == "side":
            cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
            cv.rectangle(image, (30, 180), (30 + min(side_confidence_score * 25, 175), 200),
                         (0, max(255 - side_confidence_score * 30, 50), 50), -1)
            cv.putText(image, f"Forward Head: {side_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       final_color, 1)
        else:
            # Front view confidence bars
            cv.rectangle(image, (30, 180), (30 + min(head_confidence_score * 25, 175), 200),
                         (0, max(255 - head_confidence_score * 30, 50), 50), -1)
            cv.putText(image, f"Head Issues: {head_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       final_color, 1)
            cv.rectangle(image, (30, 225), (30 + min(body_confidence_score * 25, 175), 245),
                         (0, max(255 - body_confidence_score * 30, 50), 50), -1)
            cv.putText(image, f"Body Issues: {body_confidence_score}/7", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       final_color, 1)

    # Display simplified status with score
    if calibrated_thresholds and not is_calibrating:
        if mode == "front":
            # Show both raw confidence and smoothed confidence
            raw_conf_text = f"Raw: {combined_confidence}/7"
            smooth_conf_text = f"Smooth: {smoothed_confidence:.1f}/7"
            status_text = f"{status} | {raw_conf_text} | {smooth_conf_text}"
        else:
            status_text = f"{status} (Score: {side_confidence_score}/7)"
    else:
        status_text = status
        
    cv.putText(image, status_text, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Initialize MediaPipe models and camera
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
    min_face_detection_confidence=0.5,  # Lowered from 0.7 for better detection
    min_tracking_confidence=0.5)        # Lowered from 0.7 for better detection

cap = cv.VideoCapture(0)
cv.namedWindow('Posture Detection')

with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
    with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # FIXED: Convert BGR to RGB for MediaPipe
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            timestamp = int(round(time.time() * 1000))
            pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
            face_results = face_landmarker.detect_for_video(face_image, timestamp)
            annotated_image = np.copy(frame)

            draw_landmarks(annotated_image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, drawing_styles.get_default_pose_landmarks_style())

            if pose_results.pose_landmarks:
                analyze_posture(annotated_image, pose_results.pose_landmarks[0], face_results.face_landmarks)

            cv.imshow('Posture Detection', annotated_image)

            key = cv.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                calibration_start_time = time.time()
                is_calibrating = True
                calibration_data = {k: [] for k in calibration_data}
            elif key == ord('f'):
                # Toggle debug face mesh visualization
                import builtins
                if hasattr(builtins, 'debug_face'):
                    builtins.debug_face = not builtins.debug_face
                else:
                    builtins.debug_face = True
                print(f"[DEBUG] Face Mesh visualization: {'ON' if builtins.debug_face else 'OFF'}")
            elif key == ord('d'):
                # Toggle debug cheek visualization (replaces ear debug since we use cheeks as ears)
                import builtins
                if hasattr(builtins, 'debug_cheeks'):
                    builtins.debug_cheeks = not builtins.debug_cheeks
                else:
                    builtins.debug_cheeks = True
                print(f"[DEBUG] Cheek visualization: {'ON' if builtins.debug_cheeks else 'OFF'}")
            elif key == ord('i'):
                # Print current detection info
                print(f"[INFO] Face detected: {bool(face_results.face_landmarks)}")
                print(f"[INFO] Pose detected: {bool(pose_results.pose_landmarks)}")
                if face_results.face_landmarks:
                    print(f"[INFO] Face landmarks count: {len(face_results.face_landmarks[0])}")
                print(f"[INFO] Face debug mode: {getattr(builtins, 'debug_face', False)}")
                print(f"[INFO] Calibrated: {bool(calibrated_thresholds)}")
                print(f"[INFO] Current stable posture: {current_stable_posture}")
                print(f"[INFO] Smoothed confidence: {smoothed_confidence:.2f}")

cap.release()
cv.destroyAllWindows()
log_file.close()