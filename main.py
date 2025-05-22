import cv2 as cv
import time
import mediapipe as mp
import math
import torch
import numpy as np
import threading

# Load MiDaS depth estimation model and its transformation pipeline
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Given a landmark, convert its normalized coordinates to pixel space and retrieve the corresponding depth value
# Returns None if the landmark is out of bounds

def get_depth_from_midas(landmark, image_width, image_height, depth_map):
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
        depth = depth_map[y, x]
        print(f"[DEBUG] Mapped landmark ({landmark.x:.2f}, {landmark.y:.2f}) to pixel ({x},{y}) → depth={depth:.4f}")
        return float(depth)
    else:
        print(f"[WARNING] Landmark ({landmark.x:.2f}, {landmark.y:.2f}) out of bounds at ({x},{y})")
        return None

# Compute the angle in degrees between two 2D vectors

def calculate_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle_rad = math.acos(dot / (mag1 * mag2))
    return math.degrees(angle_rad)

# Exponential smoothing function to reduce frame-to-frame jitter

def smooth(prev, current, alpha=0.2):
    return (1 - alpha) * prev + alpha * current

# Initialize MediaPipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open the webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Parameter thresholds for posture classification
# These can be customized to fit different user sensitivity levels
POSTURE_PARAMS = {
    'shoulder_tilt_threshold': 0.03,
    'z_diff_forward_threshold': 0.05,
    'head_to_shoulder_threshold': 0.08,
    'spine_bad_angle': 30,
    'spine_warn_angle': 20
}

# Variables for posture calibration
calibration_mode = False
calibration_depth_diffs = []
calibrated_depth_threshold = None
calibration_start_time = None
calibration_duration = 5
countdown_duration = 3  # Seconds to count down before calibration
hold_duration = 5       # Seconds to hold good posture

# Track which posture view mode we are analyzing: front or side
mode = "front"

# Thread-safe variables for MiDaS
cached_depth_map = None
new_frame = None
lock = threading.Lock()

# Run MiDaS continuously in a background thread

def run_midas():
    global cached_depth_map, new_frame
    while True:
        if new_frame is not None:
            with lock:
                frame = new_frame.copy()
                new_frame = None
            resized = cv.resize(frame, (256, 256))
            input_batch = transform(resized).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            cached_depth_map = prediction.to("cpu").numpy()

# Launch MiDaS in a daemon thread
threading.Thread(target=run_midas, daemon=True).start()

results = None
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    prev_slouch_angle = 0
    prev_spine_angle = 0
    prev_frame_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue

        # Pass new frame to MiDaS
        with lock:
            if new_frame is None:
                new_frame = frame.copy()

        # Resize and prepare frame for MediaPipe
        frame = cv.resize(frame, (640, 480))
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # If MiDaS hasn't returned depth yet, wait
        if cached_depth_map is None:
            cv.putText(image, "Waiting for depth...", (30, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Posture Detection", image)
            if cv.waitKey(5) & 0xFF == 27:
                break
            continue

        depth_map = cached_depth_map

        # Show live FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        cv.putText(image, f"FPS: {int(fps)}", (30, 110),
                   cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 1)

        # Full posture logic restored
        if results and results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark
            h, w, _ = image.shape

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

            depth_nose = get_depth_from_midas(nose, w, h, depth_map)
            depth_hip = get_depth_from_midas(left_hip, w, h, depth_map)

            nose_depth_display = f"{depth_nose:.2f} mm" if depth_nose is not None else "N/A"

            if depth_nose is not None and depth_hip is not None:
                z_diff_nose_hip = smooth(prev_slouch_angle, depth_nose - depth_hip)
                prev_slouch_angle = z_diff_nose_hip
            else:
                z_diff_nose_hip = 0

            world_left_shoulder = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            world_right_hip = world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            world_left_hip = world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            mid_back_world = [
                (world_left_shoulder.x + world_right_hip.x) / 2,
                (world_left_shoulder.y + world_right_hip.y) / 2
            ]
            spine_vector = [mid_back_world[0] - world_left_hip.x,
                            mid_back_world[1] - world_left_hip.y]
            vertical_vector = [0, -1]
            spine_angle = calculate_angle(spine_vector, vertical_vector)
            spine_angle = smooth(prev_spine_angle, spine_angle)
            prev_spine_angle = spine_angle

            shoulder_tilt = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y -
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            mid_eye_z = (left_eye.z + right_eye.z) / 2
            head_to_shoulder_height = abs(nose.y - ((left_shoulder.y + landmarks[mp_pose.PoseLandmark.LEFT_EAR].y) / 2))

            posture_status = "Analyzing..."
            color = (255, 255, 255)

            if calibration_mode:
                elapsed = time.time() - calibration_start_time
                if elapsed < countdown_duration:
                    remaining = int(countdown_duration - elapsed) + 1
                    cv.putText(image, f"Starting in: {remaining}s", (30, 150),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif elapsed < countdown_duration + hold_duration:
                    calibration_depth_diffs.append(z_diff_nose_hip)
                    cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    calibration_mode = False
                    calibrated_depth_threshold = sum(calibration_depth_diffs) / len(calibration_depth_diffs)
                    print(f"Calibration complete. Depth threshold set to: {calibrated_depth_threshold:.3f}")
                    cv.putText(image, "Calibration Complete", (30, 150),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif calibrated_depth_threshold is not None:
                if mode == "front":
                    if shoulder_tilt > POSTURE_PARAMS['shoulder_tilt_threshold']:
                        posture_status = "Shoulder Tilt Detected"
                        color = (0, 0, 255)
                    elif z_diff_nose_hip < calibrated_depth_threshold - POSTURE_PARAMS['z_diff_forward_threshold']:
                        posture_status = "Leaning Forward"
                        color = (0, 0, 255)
                    elif head_to_shoulder_height < POSTURE_PARAMS['head_to_shoulder_threshold']:
                        posture_status = "Head Tilt Detected"
                        color = (0, 0, 255)
                    else:
                        posture_status = "Good Front Posture"
                        color = (0, 255, 0)
                elif mode == "side":
                    if spine_angle > POSTURE_PARAMS['spine_bad_angle']:
                        posture_status = "Hunched Spine"
                        color = (0, 0, 255)
                    elif spine_angle > POSTURE_PARAMS['spine_warn_angle']:
                        posture_status = "Moderate Curve"
                        color = (0, 165, 255)
                    else:
                        posture_status = "Good Side Posture"
                        color = (0, 255, 0)

                cv.putText(image, posture_status, (30, 90),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ➕ Draw neck-to-shoulder and shoulder-to-hip angles visually like RTMPose/monitoring tools
            ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            # Vectors
            vec_neck = [ear.x - shoulder.x, ear.y - shoulder.y]
            vec_torso = [hip.x - shoulder.x, hip.y - shoulder.y]
            neck_angle = calculate_angle(vec_neck, vec_torso)

            # Draw lines
            cv.line(image, (int(ear.x * w), int(ear.y * h)), (int(shoulder.x * w), int(shoulder.y * h)), (0, 0, 255), 2)
            cv.line(image, (int(hip.x * w), int(hip.y * h)), (int(shoulder.x * w), int(shoulder.y * h)), (0, 0, 255), 2)
            # Draw angle text
            cv.putText(image, f"{int(neck_angle)}°", (int(shoulder.x * w), int(shoulder.y * h) - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Mode: {mode}", (30, 20),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
            cv.putText(image, f"Spine Angle: {round(spine_angle, 1)}", (30, 50),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
            cv.putText(image, f"Nose-Hip Δ: {z_diff_nose_hip:.2f}", (30, 70),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
            cv.putText(image, f"Nose Depth: {nose_depth_display}", (30, 140),
                       cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 255), 2)

        # Show the frame
        cv.imshow("Posture Detection", image)

        # Key press handling: ESC to quit, 'c' to calibrate, 'm' to toggle mode
        key = cv.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('c'):
            calibration_mode = True
            calibration_depth_diffs.clear()
            calibration_start_time = time.time()
            print("Calibration started. Hold good posture.")
        elif key == ord('m'):
            mode = "side" if mode == "front" else "front"
            print(f"Switched to {mode} mode")

# Release webcam and close window
cap.release()
cv.destroyAllWindows()

