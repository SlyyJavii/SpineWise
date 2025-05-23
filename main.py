import cv2 as cv
import time
import mediapipe as mp
import math
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
mode = "front"
calibration_data = {
    "slouch_angles": [],
    "z_diffs": [],
    "nose_hip_z_diffs": [],
    "eye_hip_z_diffs": [],
    "spine_angles": [],
    "sitting_heights": [],
    "head_to_shoulder_heights": []
}
calibrated_thresholds = {}

# Start speech recognition listener
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
    except Exception as e:
        print("[ERROR] Microphone failed:", e)

print("[Thread] Starting voice command thread...")
threading.Thread(target=listen_for_speech, daemon=True).start()

# The rest of the posture detection logic continues below...

# Store manually editable shoulder points (None means use MediaPipe values)
manual_left_shoulder = None
manual_right_shoulder = None
editing_shoulder = None  # Track which shoulder is being edited ("left" or "right")

# Mouse callback function to allow clicking and dragging shoulder points
def mouse_callback(event, x, y, flags, param):
    global manual_left_shoulder, manual_right_shoulder, editing_shoulder
    if event == cv.EVENT_LBUTTONDOWN:
        if manual_left_shoulder and abs(manual_left_shoulder[0] - x) < 20 and abs(manual_left_shoulder[1] - y) < 20:
            editing_shoulder = "left"
        elif manual_right_shoulder and abs(manual_right_shoulder[0] - x) < 20 and abs(manual_right_shoulder[1] - y) < 20:
            editing_shoulder = "right"
    elif event == cv.EVENT_MOUSEMOVE and editing_shoulder:
        if editing_shoulder == "left":
            manual_left_shoulder = (x, y)
        elif editing_shoulder == "right":
            manual_right_shoulder = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        editing_shoulder = None

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

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark

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
            h, w, _ = image.shape

            if manual_left_shoulder:
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

            mid_back = [(left_shoulder.x + right_hip.x) / 2,
                        (left_shoulder.y + right_hip.y) / 2]

            world_left_shoulder = world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            world_right_hip = world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            world_left_hip = world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            mid_back_world = [
                (world_left_shoulder.x + world_right_hip.x) / 2,
                (world_left_shoulder.y + world_right_hip.y) / 2
            ]

            spine_vector = [mid_back_world[0] - world_left_hip.x, mid_back_world[1] - world_left_hip.y]
            spine_angle = calculate_angle(spine_vector, vertical_vector)
            spine_angle = smooth(prev_spine_angle, spine_angle)
            prev_spine_angle = spine_angle

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Spine Angle: {round(spine_angle, 1)} deg", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

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


