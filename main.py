import cv2 as cv 
import time
import mediapipe as mp
import math 

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calibration variables
is_calibrating = False
calibration_data = {
    "slouch_angles": [],
    "z_diffs": [],
    "nose_hip_z_diffs": [],
    "eye_hip_z_diffs": []
}
calibrated_thresholds = {}

# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle_rad = math.acos(dot / (mag1 * mag2))
    return math.degrees(angle_rad)

# Open webcam
cap = cv.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    mode = "front"  # Default mode
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue

        # Default posture status
        posture_status = "No pose detected"
        color = (128, 128, 128)

        # Convert color and prepare image
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get landmarks
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

            # Adjust shoulder position using ear height
            adjusted_left_shoulder_y = (left_shoulder.y + left_ear.y) / 2
            adjusted_left_shoulder = [left_shoulder.x, adjusted_left_shoulder_y]

            # Calculate torso vector and vertical reference
            torso_vector = [
                adjusted_left_shoulder[0] - left_hip.x,
                adjusted_left_shoulder[1] - left_hip.y
            ]
            vertical_vector = [0, -1]

            # Compute slouch angle and Z-depth difference
            slouch_angle = calculate_angle(torso_vector, vertical_vector)
            z_diff = left_shoulder.z - left_hip.z

            # ✅ Store calibration data *after* computing metrics
            if is_calibrating:
                calibration_data["slouch_angles"].append(slouch_angle)
                calibration_data["z_diffs"].append(z_diff)
                calibration_data["nose_hip_z_diffs"].append(z_diff_head_nose)
                calibration_data["eye_hip_z_diffs"].append(z_diff_head_eye)
                cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Classify posture based on current mode and thresholds
            if calibrated_thresholds and mode == "front":
                if slouch_angle > calibrated_thresholds["slouch_bad"] or z_diff < calibrated_thresholds["z_bad"] or z_diff_head_nose < calibrated_thresholds["nose_bad"]:
                    posture_status = "Slouching!"
                    color = (0, 0, 255)
                elif slouch_angle > calibrated_thresholds["slouch_warn"] or z_diff < calibrated_thresholds["z_warn"] or z_diff_head_nose < calibrated_thresholds["nose_warn"]:
                    posture_status = "Moderate Slouch"
                    color = (0, 165, 255)
                else:
                    posture_status = "Great Posture!"
                    color = (0, 255, 0)

            elif calibrated_thresholds and mode == "side":
                if z_diff_head_nose < calibrated_thresholds["nose_bad"] or z_diff_head_eye < calibrated_thresholds["eye_bad"]:
                    posture_status = "Hunched Forward"
                    color = (0, 0, 255)
                elif z_diff_head_nose < calibrated_thresholds["nose_warn"]:
                    posture_status = "Moderate Forward Lean"
                    color = (0, 165, 255)
                else:
                    posture_status = "Good Side Posture"
                    color = (0, 255, 0)

            # Draw landmarks and posture info
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, posture_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Debug print
            print("Slouch angle:", slouch_angle, "| Nose Z diff:", z_diff_head_nose)

        # Show webcam image
        cv.imshow('Posture Detection', image)

        # Handle key inputs
        key = cv.waitKey(5) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('m'):
            mode = "side" if mode == "front" else "front"
            print("Switched to", mode)
        elif key == ord('c'):
            is_calibrating = not is_calibrating
            if is_calibrating:
                print("Calibration started. Hold good posture.")
                calibration_data = {k: [] for k in calibration_data}  # Reset
            else:
                print("Calibration ended. Processing...")
                if len(calibration_data["slouch_angles"]) > 0:
                    avg_angle = sum(calibration_data["slouch_angles"]) / len(calibration_data["slouch_angles"])
                    avg_z_diff = sum(calibration_data["z_diffs"]) / len(calibration_data["z_diffs"])
                    avg_nose = sum(calibration_data["nose_hip_z_diffs"]) / len(calibration_data["nose_hip_z_diffs"])
                    avg_eye = sum(calibration_data["eye_hip_z_diffs"]) / len(calibration_data["eye_hip_z_diffs"])

                    calibrated_thresholds = {
                        "slouch_warn": avg_angle + 10,
                        "slouch_bad": avg_angle + 20,
                        "z_warn": avg_z_diff - 0.05,
                        "z_bad": avg_z_diff - 0.15,
                        "nose_warn": avg_nose - 0.1,
                        "nose_bad": avg_nose - 0.25,
                        "eye_warn": avg_eye - 0.1,
                        "eye_bad": avg_eye - 0.25
                    }
                    print("Calibration complete.")
                    print("Thresholds:", calibrated_thresholds)

# Cleanup
cap.release()
cv.destroyAllWindows()




