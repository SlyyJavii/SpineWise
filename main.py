import cv2 as cv 
import mediapipe as mp
import math 

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

            # Classify posture based on current mode
            if mode == "front":
                if slouch_angle > 35 or z_diff < -0.2:
                    posture_status = "Slouching!"
                    color = (0, 0, 255)
                elif slouch_angle > 25 or z_diff < -0.12:
                    posture_status = "Moderate Slouch"
                    color = (0, 165, 255)
                elif slouch_angle > 15:
                    posture_status = "Slight Lean"
                    color = (0, 255, 255)
                else:
                    posture_status = "Great Posture!"
                    color = (0, 255, 0)
            else:  # side view
                if z_diff < -0.25:
                    posture_status = "Hunched Forward"
                    color = (0, 0, 255)
                elif z_diff < -0.15:
                    posture_status = "Moderate Forward Lean"
                    color = (0, 165, 255)
                elif z_diff < -0.07:
                    posture_status = "Slight Lean"
                    color = (0, 255, 255)
                else:
                    posture_status = "Good Side Posture"
                    color = (0, 255, 0)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display posture info
            cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(image, posture_status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Debugging prints
            print("Left shoulder:", left_shoulder.x, left_shoulder.y, left_shoulder.z)
            print("Right shoulder:", right_shoulder.x, right_shoulder.y, right_shoulder.z)
            print("Left hip:", left_hip.x, left_hip.y, left_hip.z)
            print("Right hip:", right_hip.x, right_hip.y, right_hip.z)

        # Show the image
        cv.imshow('Posture Detection', image)

        # Handle key inputs
        key = cv.waitKey(5) & 0xFF
        if key == 27:  # ESC key to exit
            break
        if key == ord('m'):  # Toggle posture detection mode
            mode = "side" if mode == "front" else "front"
            print("Switched to", mode)

# Release webcam and close window
cap.release()
cv.destroyAllWindows()

            



