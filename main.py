# This is the main.py of the (ongoing) SpineWise posture rating system application project for Capstone 1 and 2.
# This code utilizes MediaPipe and OpenCV to track landmarks on the human body and assess the quality of posture
# from both front and side views. This currently encompasses the base backend of the code.
# As of right now, this code is... almost up to date? I say this because John and Javi updates their codes often
# and this code might become more and more behind as a result. Any major changes that aren't fine-tuning could just
# batter this code in functionality.

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
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QTabWidget, QCheckBox, QGroupBox, \
    QFormLayout, QSpinBox, QSlider, QGridLayout, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

# Global Flag Dump. This'll be a baseline place to put settings-adjusted features/variables in.
facelandtoggle = True
poselandtoggle = True
camera_active = True
notification_volume = 50  # FOR JAKE AND EMDYA: This is the placeholder for the notification system volume. I don't know how it'll work with your own code set-ups, but adjust accordingly. Thanks! :]


# Pose models. This downloads them if they aren't present!
pose_model = "pose_landmarker_full.task"
if not exists(pose_model):
    pose_model = urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "pose_landmarker_full.task")[0]

face_model = "face_landmarker.task"
if not exists(face_model):
    face_model = urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "face_landmarker.task")[0]

# Open a CSV file that logs timestamp of latest entry.
# This generates a CSV with a timestamp, mode parameter, facing, posture status,
# head tilt, and confidence score. (Note: Juan has never opened these CSVs, so I don't know how they look.)
last_log_time = time.time()
log_file = open("posture_trend_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])

# Calibration calls.
is_calibrating = False
calibration_data = {
    "facial_distances": [], # ASSUMING this is face-to-camera depth for angle calculation against torso distance.
    "torso_distances": [], # Ditto.
    "clavicle_lengths": [], # Shoulder-to-shoulder distance. This aids in the later mode-switch part of the code.
    "face_torso_heights": [], # Height from face-to-torso.
    "shoulder_ear_distance": []
}
countdown_duration = 3 # Countdown.
hold_duration = 5 # Holds for 5 seconds; will most likely be configurable in the future since it gets annoying to sit for 5 seconds.
calibrated_thresholds = {} # Storage of calibration thresholds.
calibration_start_time = 0

mode = "front" # Initial mode assumes user is facing front when program opens. Something to look into with new MP.
prev_facial_distances = [0, 0, 0, 0, 0, 0] # This makes a mean of 6 previous distances of their corresponding variables.
prev_torso_distances = [0, 0, 0, 0, 0, 0] # Ditto.
cache_idx = 0 # This indexes into circular buffer.

status_enum = (
    ("Good Posture", (0, 220, 0)),
    ("Early Posture Warning", (0, 220, 0)),
    ("Significant Postural Issue", (0, 220, 0)),
    ("Severe Slouch + Lean", (0, 0, 255)),
    ("Critical Forward Posture", (128, 0, 0))
)

# Basic MediaPipe posing and drawing calls.
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
drawing_connections = mp.solutions.pose.POSE_CONNECTIONS
default_face_connections_style = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

# NOTE: We might need to fiddle with the microphone functionality a little. There's possibility for Win32 thread conflicts
# With PyQt5 drawing the application. It MIGHT need to start running AFTER the application is drawn. Not entirely sure yet.

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

    left_shoulder_ear = np.linalg.norm(
        np.array((left_shoulder.x, left_shoulder.y)) - np.array((left_ear.x, left_ear.y)))
    right_shoulder_ear = np.linalg.norm(
        np.array((right_shoulder.x, right_shoulder.y)) - np.array((right_ear.x, right_ear.y)))
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
            cv.putText(image, "CALIBRATING... Hold Good Posture", (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        else:
            is_calibrating = False
            calibrated_thresholds = {
                "clavicle_length_threshold": avg("clavicle_lengths") * 0.7
            }

    if calibrated_thresholds and not is_calibrating:
        mode = "side" if clavicle_length < calibrated_thresholds["clavicle_length_threshold"] else "front"
        if mode == "front":
            facial_avg = avg("facial_distances")
            torso_avg = avg("torso_distances")
            face_clav_height_avg = avg("face_torso_heights")
            shoulder_ear_avg = avg("shoulder_ear_distance")
            shoulder_ear_percentage = (avg_shoulder_ear - shoulder_ear_avg) / shoulder_ear_avg

            facial_percentage = (facial_distance - facial_avg) / facial_avg
            torso_percentage = (torso_distance - torso_avg) / torso_avg
            height_percentage = (face_torso_height - face_clav_height_avg) / face_clav_height_avg

            head_confidence_score += math.floor(abs(facial_percentage) / 0.15)
            head_confidence_score += math.floor(abs(head_tilt_difference) / 0.075)
            head_confidence_score = min(7, max(head_confidence_score, 0))

            body_confidence_score += math.floor(abs(facial_percentage - torso_percentage) / 0.125)
            body_confidence_score += math.floor(abs(height_percentage) / 0.1)
            body_confidence_score += math.floor(abs(shoulder_ear_percentage) / 0.1)
            body_confidence_score = min(7, max(body_confidence_score, 0))

            combined_confidence = math.floor((head_confidence_score + body_confidence_score) / 2) - 1
            if combined_confidence < 1:
                combined_confidence = 0
            if combined_confidence > 4:
                combined_confidence = 4

            status_idx = status_enum[combined_confidence]
            status = status_idx[0]
            color = status_idx[1]


        elif mode == "side":

            # Determine which shoulder is closer to screen center
            facing = "left" if left_shoulder.x > right_shoulder.x else "right"
            if facing == "left":
                shoulder = right_shoulder
                hip = right_hip
                side_label = "Right Side View"
            else:
                shoulder = left_shoulder
                hip = left_hip
                side_label = "Left Side View"

            # Calculate a side-specific slouch angle
            slouch_vector = [shoulder.x - hip.x, shoulder.y - hip.y]
            slouch_angle = calculate_angle(slouch_vector, [0, -1])

    average_color = image[30:310, 175:220].mean((0, 1))
    final_color = ((255 - average_color[0]), (255 - average_color[1]), (255 - average_color[2]))

    cv.putText(image, f"Mode: {mode}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (final_color[0], 220, final_color[2]), 2)

    if mode == "side":
        cv.putText(image, side_label, (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
        cv.putText(image, f"Slouch Angle: {round(slouch_angle, 1)} deg", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)

    cv.putText(image, status, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv.rectangle(image, (30, 180), (30 + head_confidence_score * 40, 200), (0, 255 - head_confidence_score * 50, 50),
                 -1)
    cv.putText(image, f"Head Confidence: {head_confidence_score}/7", (30, 175), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               final_color, 1)
    cv.rectangle(image, (30, 225), (30 + body_confidence_score * 40, 245), (0, 255 - body_confidence_score * 50, 50),
                 -1)
    cv.putText(image, f"Body Confidence: {body_confidence_score}/7", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               final_color, 1)

# BIG BIG BLOCK SURROUNDING THE APPLICATION ITSELF. This was a nightmare to make. Really fun! Really hectic.
# Essentially, utilizing PyQt5 renders a lot of our old OpenCV window-related code moot. This meant a trial and error approach of putting in
# Qt modules that served the purposes that OpenCV served in rendering that old window, and YANKING out old OpenCV code since it served no more purpose.
# This utilized PyQt5 with *some* aspects of OpenCV, mainly just relegating it to doing computer vision and not much else. In fact, some of these blocks rip code out
# FROM our old OpenCV blocks and insert it into the PyQt5 blocks with slight adjustment.
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._camera_active = True  # Internal camera state

        base_pose_options = python.BaseOptions(model_asset_path=pose_model)
        self.pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_pose_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        base_face_options = python.BaseOptions(model_asset_path=face_model)
        self.face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_face_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def set_camera_active(self, active):
        """Method to control camera state from main thread"""
        self._camera_active = active

    def run(self):
        cap = None

        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(self.pose_options) as pose_landmarker:
                with mp.tasks.vision.FaceLandmarker.create_from_options(self.face_options) as face_landmarker:
                    while self._run_flag:
                        if self._camera_active:
                            # Initialize camera if it's not already open
                            if cap is None:
                                cap = cv.VideoCapture(0)
                                if not cap.isOpened():
                                    print("[VIDEO] Could not open camera")
                                    continue

                            success, frame = cap.read()
                            if not success:
                                continue

                            try:
                                timestamp = int(round(time.time() * 1000))
                                pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
                                face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))

                                pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
                                face_results = face_landmarker.detect_for_video(face_image, timestamp)
                                annotated_image = np.copy(frame)

                                # Only draw pose landmarks if the global flag is True.
                                if pose_results.pose_landmarks and poselandtoggle:
                                    draw_landmarks(
                                        annotated_image,
                                        pose_results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS,
                                        drawing_styles.get_default_pose_landmarks_style()
                                    )
                                    analyze_posture(annotated_image, pose_results.pose_landmarks[0])
                                else:
                                    # Still analyze posture even if we don't draw the skeleton.
                                    if pose_results.pose_landmarks:
                                        analyze_posture(annotated_image, pose_results.pose_landmarks[0])

                                # Only draw face landmarks if the global flag is True.
                                if face_results.face_landmarks and facelandtoggle:
                                    draw_landmarks(
                                        annotated_image,
                                        face_results.face_landmarks,
                                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                        drawing_styles.DrawingSpec((255, 255, 255), 1, 1)
                                    )

                                self.change_pixmap_signal.emit(annotated_image)

                            except Exception as e:
                                print("[VIDEO] Frame‐processing error:", e)
                                continue
                        else:
                            # Camera is disabled - release resources and emit blank frame
                            if cap is not None:
                                cap.release()
                                cap = None

                            # Emit a blank/black frame
                            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv.putText(blank_frame, "Camera Disabled", (200, 240),
                                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            self.change_pixmap_signal.emit(blank_frame)

                            # Sleep a bit to prevent busy waiting
                            time.sleep(0.1)

        except Exception as e:
            print("[VIDEO] Initialization error:", e)
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# Actual app content. Emdya or Javi, whichever one gets around to reading this first, could you possibly
# get the SpineWise logo to be our App icon? I'm not particular sure how that would work, I'm assuming we would
# grab the file as an .ico file and then uses it for our top left icon and our taskbar icon.
# I just think it'd be fun and a LOT better than the usual placeholder icons.
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpineWise [TM] (w.i.p.)")
        self.resize(1000, 600)

        # This builds up the tab lay-out as base.
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

        # Our three tabs. With this in mind, IN DESCENDING ORDER FOR SORTING, new tabs can be added and sorted.
        self.camera_tab = QWidget()
        self.analytics_tab = QWidget()
        self.settings_tab = QWidget()
        self.tab_widget.addTab(self.camera_tab, "Dashboard (or just camera right now)")
        self.tab_widget.addTab(self.analytics_tab, "Analytics")
        self.tab_widget.addTab(self.settings_tab, "Settings")

        # This is for later building of the tabs in their own defs.
        self._build_camera_tab()
        self._build_analytics_tab()
        self._build_settings_tab()

        # This runs the video thread that was made way above. See (what should be if not edited crazily) line 345.
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image_in_label)
        self.thread.start()

    def closeEvent(self, event): # Handles closing thread when app is closed. Also closes the CSV.
        self.thread.stop()
        try:
            log_file.close()
        except Exception:
            pass
        event.accept()

        # Tab that currently holds our camera. This is going to be our dashboard later on.
    def _build_camera_tab(self):
        grid = QGridLayout()

        cam_group = QGroupBox("Camera Feed")
        cam_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        cam_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        self.toggle_btn = QPushButton("Disable Camera")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self._toggle_camera)
        cam_layout.addWidget(self.toggle_btn, alignment=Qt.AlignCenter)
        cam_group.setLayout(cam_layout)

        status_group = QGroupBox("Session Status")
        status_layout = QVBoxLayout()
        self.timer_label = QLabel("Session Time: 00:00:00")
        status_layout.addWidget(self.timer_label)
        self.camera_status = QLabel("Camera Active: Yes")
        status_layout.addWidget(self.camera_status)
        status_group.setLayout(status_layout)

        grid.addWidget(cam_group, 0, 0)
        grid.addWidget(status_group, 0, 1)
        self.camera_tab.setLayout(grid)

        from PyQt5.QtCore import QTimer
        self.session_start = time.time()
        self.qtimer = QTimer()
        self.qtimer.timeout.connect(self._update_timer)
        self.qtimer.start(1000)

    def _update_timer(self):
        elapsed = int(time.time() - self.session_start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.timer_label.setText(f"Session Time: {h:02d}:{m:02d}:{s:02d}")

    def _toggle_camera(self, checked):
        global camera_active
        camera_active = not checked

        # Update the video thread's camera state
        self.thread.set_camera_active(camera_active)

        # Update UI elements
        self.camera_status.setText(f"Camera Active: {'Yes' if camera_active else 'No'}")
        self.toggle_btn.setText('Enable Camera' if checked else 'Disable Camera')

        print(f"[DEBUG] Camera toggled: {'OFF' if checked else 'ON'}")

    def update_camera_view(self, image):
        qt_img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    # Placeholder analytics tab. See Emdya's Design 2 mock-up that had some pretty good ideas for the analytics tab.
    # Real time posture plot. Holy shit this actually sucks. None of this looks particularly good or works particularly well.
    # This is the one part of the code I hate. Hate hate hate. If this doesn't look updated, I most likely reverted all of this to base placeholder.
    def _build_analytics_tab(self):
        analytics_layout = QVBoxLayout()
        placeholder = QLabel("analytics'll be here soon (maybe)")
        placeholder.setStyleSheet("font-size: 18px; color: gray;")
        analytics_layout.addWidget(placeholder, alignment=Qt.AlignCenter)
        self.analytics_tab.setLayout(analytics_layout)

    # jingle jingle
    # this was me fiddling around with the sliders and knobs like a total mook but essentially
    # It's the Settings tab. Self-explanatory. Right now it has very little, but will have more for when notifications are integrated!
    def _build_settings_tab(self):
        settings_layout = QVBoxLayout()

        # Block for Overlays! This is where the landmarks go. Will probably allow for toggling of other metrics.
        overlays_group = QGroupBox("Overlays")
        overlays_layout = QFormLayout()

        # Pose landmarks handling.
        self.pose_checkbox = QCheckBox("Show Pose Landmarks")
        self.pose_checkbox.setChecked(poselandtoggle)
        self.pose_checkbox.stateChanged.connect(self._on_pose_toggled)
        overlays_layout.addRow(self.pose_checkbox)

        # Ditto but for the face.
        self.face_checkbox = QCheckBox("Show Face Landmarks")
        self.face_checkbox.setChecked(facelandtoggle)
        self.face_checkbox.stateChanged.connect(self._on_face_toggled)
        overlays_layout.addRow(self.face_checkbox)

        overlays_group.setLayout(overlays_layout)
        settings_layout.addWidget(overlays_group)

        # Block for Notifications! This will go for notifications. Right now, this is just a placeholder with a fun slide.
        # Which means! JAKE AND EMDYA this is yours to adjust.
        notif_group = QGroupBox("Notification Settings")
        notif_layout = QHBoxLayout()

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setTickPosition(QSlider.TicksBelow)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.setValue(notification_volume)
        self.volume_slider.valueChanged.connect(self._on_notif_slider_changed)
        self.volume_label = QLabel(f"{notification_volume}%")
        notif_layout.addWidget(self.volume_slider)
        notif_layout.addWidget(self.volume_label)
        notif_group.setLayout(notif_layout)
        settings_layout.addWidget(notif_group)

        # Block for Calibration Time! In the future, will probably only get one more setting.
        # AKA the setting to adjust how LONG the countdown takes.
        calib_group = QGroupBox("Calibration Time")
        calib_layout = QFormLayout()

        # Calibration can now be adjusted *down* from 5 seconds. This is great because sitting there for 5 seconds
        # was a little long. However, I *don't* know how badly that may influence calibration data.
        self.hold_spin = QSpinBox()
        self.hold_spin.setRange(1, 20)
        self.hold_spin.setValue(hold_duration)
        self.hold_spin.valueChanged.connect(self._on_hold_duration_changed)
        calib_layout.addRow("Hold Duration (in seconds) [Note: May change accuracy of calibration data! You have been warned!]:", self.hold_spin)

        calib_group.setLayout(calib_layout)
        settings_layout.addWidget(calib_group)

        # Fill vertical space at bottom
        settings_layout.addStretch()
        self.settings_tab.setLayout(settings_layout)

    # hey remember our old keypress stuff on opencv where escape was 27 and it was odd
    # Yeah now it's Escape.
    # These are the same keypress keys as before. C for calib, ESC for Escape. Now handled by Qt!
    def keyPressEvent(self, event):
        global is_calibrating, calibration_data, calibration_start_time
        if event.key() == Qt.Key_C:
            calibration_start_time = time.time()
            is_calibrating = True
            calibration_data = {k: [] for k in calibration_data}
            print("Calibrating. Please wait...")
        elif event.key() == Qt.Key_Escape:
            self.close()

    # SLOTS AND HELPERS IF YOU CHANGE *ANYTHING* IN THE SETTINGS, YOU MUST PUT A HELPER IN HERE.
    def _on_pose_toggled(self, state):
        global poselandtoggle
        poselandtoggle = (state == Qt.Checked)

    def _on_face_toggled(self, state):
        global facelandtoggle
        facelandtoggle = (state == Qt.Checked)

    def _on_hold_duration_changed(self, val):
        global hold_duration
        hold_duration = val
        print(f"[Settings Tab] hold_duration set to {hold_duration}s")

    # FROM JUAN TO JAKE AND EMDYA: This is *also* for notifications. Adjust accordingly.
    def _on_notif_slider_changed(self, val):
        global notification_volume
        notification_volume = val
        self.volume_label.setText(f"{notification_volume}%")
    # In case you want to debug this setting, unhashtag this. It will ACTUALLY spam the shit out of your console.
    #   print(f"[Settings] notification_volume = {notification_volume}%")

    # This is the little baby code that takes OpenCV frame data and slaps it into PyQt5.
    @pyqtSlot(np.ndarray)
    def update_image_in_label(self, cv_img): # Convert the cv_img (BGR) to QPixmap and set it on self.video_label.
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = qt_image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

# Application entry point.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
