# Enhanced spinewise_gui.py with speech recognition integration

import os
import queue

import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import threading
import backend
import speech_recognition as sr


from PyQt5.QtWidgets import (
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget, QTabWidget, QMainWindow,QFrame,QVBoxLayout,
    QFileDialog, QTextEdit, QDoubleSpinBox, QScrollArea, QSpinBox,QHBoxLayout, QCheckBox, QFormLayout,QSlider,QGroupBox, QProgressBar, QTableWidgetItem,QTableWidget,QHeaderView
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPixmap, QIcon, QFontDatabase, QPalette, QBrush, QPixmap, QPainter
from PyQt5.QtCore import Qt, QThread, QSize, pyqtSignal, QEvent, QTimer
from backend import (
    analyze_posture, get_pose_landmarker, get_face_landmarker,
    draw_landmarks, normalize_lighting, is_calibrating,
    calibration_start_time, calibration_data, set_gui_mode
)


class SpeechRecognitionThread(QThread):
    """Thread for continuous speech recognition"""
    command_detected = pyqtSignal(str)  # Signal to emit recognized commands
    status_update = pyqtSignal(str)  # Signal for status updates

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.listening_enabled = False
        self.recognizer = None
        self.microphone = None
        self.show_landmarks = False

    
        
    def run(self):
        """Main speech recognition loop"""
        try:
            self.recognizer = sr.Recognizer()

            # Try to find the best microphone
            mic_list = sr.Microphone.list_microphone_names()
            print(f"[SPEECH] Available microphones: {mic_list}")

            # Use default microphone
            self.microphone = sr.Microphone()

            # Much more aggressive settings for better pickup
            self.recognizer.energy_threshold = 100  # Even lower threshold
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5  # Shorter pause detection
            self.recognizer.phrase_threshold = 0.2  # More sensitive phrase detection
            self.recognizer.non_speaking_duration = 0.3  # Shorter non-speaking detection

            # Adjust for ambient noise once at startup
            print("[SPEECH] Adjusting for ambient noise...")
            self.status_update.emit("Calibrating microphone...")

            with self.microphone as source:
                # Much shorter adjustment to preserve low threshold
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                # Force lower threshold if adjustment made it too high
                if self.recognizer.energy_threshold > 300:
                    self.recognizer.energy_threshold = 200
                print(f"[SPEECH] Final energy threshold: {self.recognizer.energy_threshold}")

            print("[SPEECH] Speech recognition ready")
            self.status_update.emit("Ready - try saying 'start'")

            while self._run_flag:
                if self.listening_enabled:
                    try:
                        # Listen for audio with timeout
                        with self.microphone as source:
                            print("[SPEECH] Listening... (say 'start', 'stop', 'calibrate', or 'exit')")
                            self.status_update.emit(" Listening...")
                            
                            # Even longer timeout and phrase limit for better capture
                            audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                            
                        print("[SPEECH]  Audio captured, processing...")
                        self.status_update.emit("🔄 Processing...")

                        # Try multiple recognition methods for better accuracy
                        command = None
                        try:
                            # Primary: Google Speech Recognition
                            command = self.recognizer.recognize_google(audio, language='en-US').lower().strip()
                            print(f"[SPEECH]  Google recognized: '{command}'")
                        except:
                            try:
                                # Fallback: Google with different settings
                                command = self.recognizer.recognize_google(audio, language='en', show_all=False).lower().strip()
                                print(f"[SPEECH] Google fallback recognized: '{command}'")
                            except:
                                print("[SPEECH]  Both Google recognition attempts failed")
                                continue

                        if command:
                            # Emit the recognized command
                            self.command_detected.emit(command)
                            self.status_update.emit(f"✅ Heard: '{command}'")

                    except sr.WaitTimeoutError:
                        # Timeout is normal, just continue
                        print("[SPEECH] ⏰ Listening timeout (normal)")
                        continue
                    except sr.UnknownValueError:
                        # Could not understand audio
                        print("[SPEECH]  Could not understand audio")
                        self.status_update.emit(" Could not understand - try again")
                        continue
                    except sr.RequestError as e:
                        print(f"[SPEECH]  API error: {e}")
                        self.status_update.emit(f" Speech API error: Check internet")
                        time.sleep(5)  # Wait before retrying
                        continue
                    except Exception as e:
                        print(f"[SPEECH]  Unexpected error: {e}")
                        continue
                else:
                    # Not listening, sleep briefly
                    time.sleep(0.5)

        except Exception as e:
            print(f"[SPEECH] Failed to initialize: {e}")
            self.status_update.emit(f" Mic initialization failed: {e}")
    
    def enable_listening(self):
        """Enable speech recognition"""
        self.listening_enabled = True
        print("[SPEECH] Speech recognition enabled")

    def disable_listening(self):
        """Disable speech recognition"""
        self.listening_enabled = False
        print("[SPEECH] Speech recognition disabled")

    def stop(self):
        """Stop the speech recognition thread"""
        self._run_flag = False
        self.listening_enabled = False
        self.wait()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_stats_signal = pyqtSignal(str)

    def __init__(self, show_landmarks = False):
        super().__init__()
        self._run_flag = True  # Make sure this is always True when created
        self.pose_landmarker = None
        self.face_landmarker = None
        self.raw_queue = None
        self.processed_queue = None
        self.show_landmarks = show_landmarks
        print("[VIDEO] VideoThread initialized with _run_flag = True")

    def set_landmark_visibility(self, show_landmarks):
        self.show_landmarks = show_landmarks

    def process_image_queue(self):
        with self.pose_landmarker as pose_landmarker, self.face_landmarker as face_landmarker:
            while self._run_flag:
                frame = self.raw_queue.get()

                frame = normalize_lighting(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe image objects
                timestamp = int(round(time.time() * 1000))
                pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Get results
                pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
                face_results = face_landmarker.detect_for_video(face_image, timestamp)

                # Create a copy for annotation
                annotated_image = np.copy(frame)

                if pose_results.pose_landmarks:
                    # Don't draw landmarks - keep clean video feed for GUI
                    # draw_landmarks(annotated_image, pose_results.pose_landmarks)
                     #draw landmarks only if setting is enabled
                    if self.show_landmarks:
                        draw_landmarks(annotated_image, pose_results.pose_landmarks)

                    # Analyze posture (this still works without drawing landmarks)
                    result = analyze_posture(
                        annotated_image,
                        pose_results.pose_landmarks[0],
                        face_results.face_landmarks if face_results.face_landmarks else None
                    )
                    self.update_stats_signal.emit(result)
                else:
                    self.update_stats_signal.emit("No pose detected")

                # Convert to RGB for Qt (now shows clean video without landmarks)
                rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # self.change_pixmap_signal.emit(qt_image)
                self.processed_queue.put(qt_image)

    def run(self):
        print(f"[INFO] VideoThread started with _run_flag = {self._run_flag}")

        # Create landmarkers
        self.pose_landmarker = get_pose_landmarker()
        self.face_landmarker = get_face_landmarker()
        self.raw_queue = queue.Queue()
        self.processed_queue = queue.Queue()

        threading.Thread(target=self.process_image_queue).start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Set camera properties for better quality
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"[VIDEO] About to enter main loop with _run_flag = {self._run_flag}")

        try:

            while self._run_flag:
                ret, frame = cap.read()
                if not ret:
                    print("[VIDEO] Failed to read frame")
                    continue

                self.raw_queue.put(frame)

                processed_frame = self.processed_queue.get()
                self.change_pixmap_signal.emit(processed_frame)

            print(f"[VIDEO] Exited main loop with _run_flag = {self._run_flag}")

        except Exception as e:
            print(f"[VIDEO] Exception in video thread: {e}")
        finally:
            print("[VIDEO] Releasing camera...")
            cap.release()
            print("[VIDEO] Camera released, thread ending normally")

    def stop(self):
        """Stop the video thread gracefully without closing app"""
        print("[VIDEO] Stop method called...")
        self._run_flag = False
        print("[VIDEO] Run flag set to False")

        # Don't call wait() here - let the main thread handle it
        print("[VIDEO] Stop method complete (no wait called)")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        font_path = os.path.join(os.path.dirname(__file__), "assets/fonts/PressStart2P-Regular.ttf")
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("❌ Failed to load pixel font")
        else:
            loaded_font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            print(f"✅ Loaded font: {loaded_font_family}")
        
        # CRITICAL: Set GUI mode to prevent backend speech recognition conflicts
        set_gui_mode(True)

        self.setWindowTitle("SpineWise Posture App - With Voice Control")
        self.setGeometry(100, 100, 1400, 900)

        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.tab_widget.setMinimumHeight(60)
        # Create a frame for the white tab background
        self.tab_widget.setMinimumHeight(80)
        tab_container = QFrame()
        tab_container.setStyleSheet("""
            QFrame {
                    background-color: white;
                    border-radius: 20px;
                    padding: 15px;padding: 15px;
                    margin-left: 100px;
                    margin-right: 100px;
                    }
        """)
        # Old Layout
        # Layout inside the white container
        #tab_layout = QVBoxLayout(tab_container)
        #tab_layout.addWidget(self.tab_widget)
        #tab_layout.setAlignment(Qt.AlignHCenter)
       # tab_layout.setContentsMargins(0, 0, 0, 0)

        # Outer layout for the whole window
        #main_layout = QVBoxLayout()
       # main_layout.addWidget(tab_container)  
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.tab_widget)
  

        # Set as central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)   

        background_path = os.path.join(os.path.dirname(__file__), "assets/sky_background.png")
        self.bg_pixmap = QPixmap(background_path)

        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(self.bg_pixmap.scaled(
            self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setAutoFillBackground(True)
        self.setPalette(palette)             

        # Tabs
        self.live_tab = QWidget()
        self.live_tab.setAttribute(Qt.WA_StyledBackground, True)
        self.live_tab.setStyleSheet("background-color: transparent;")
        self.log_tab = QWidget()
        self.settings_tab = QWidget()

        self.tab_widget.addTab(self.live_tab, "Live Posture")
        self.tab_widget.addTab(self.log_tab, "Posture Log")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.setStyleSheet("""
            QTabBar {
                background: transparent;
                padding: 10px;
            }

            QTabBar::tab {
                app.setFont(QFont(loaded_font_family, 10))
                background-color: #ffffff;
                border: 3px solid #000000;
                border-bottom: 4px solid #000000;
                border-right: 4px solid #000000;
                padding: 6px 20px;
                margin: 6px;
                color: black;
                font-size: 12px;
                font-family: "Press Start 2P", monospace; 
                font-weight: normal;
                min-width: 180px;
                min-height: 30px;
                text-align: center;
            }

            QTabBar::tab:selected {
                background-color: #00ff00;
            }

            QTabBar::tab:hover {
                background-color: #ccffcc;
                
            }

            QTabWidget::pane {
                border: 2px solid #000;
                background-color: transparent;
                top: -1px;
                }
            """)



        # Initialize threads
        self.show_landmarks = False
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_stats_signal.connect(self.update_stats)


        self.speech_thread = SpeechRecognitionThread()
        self.speech_thread.command_detected.connect(self.handle_voice_command)
        self.speech_thread.status_update.connect(self.update_speech_status)

        self.notification_volume = 50
        self.beep_interval = 2.0
        self.alert_duration = 10.0
        

        # Then initialize tabs
        self.init_live_tab()
        self.init_log_tab()
        self.init_settings_tab()



       
        # Start speech recognition thread
        self.speech_thread.start()

    def resizeEvent(self, event):
        palette = self.palette()
        scaled = self.bg_pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette.setBrush(QPalette.Window, QBrush(scaled))
        self.setPalette(palette)
        super().resizeEvent(event)
    


    def init_live_tab(self):
       # Create a transparent wrapper widget
        live_wrapper = QWidget()
        live_wrapper.setAttribute(Qt.WA_StyledBackground, True)
        live_wrapper.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(live_wrapper)
        pixel_font = QFont("Press Start 2P", 10)
        pixel_font.setStyleStrategy(QFont.NoAntialias)

        # Create folder QLabel
        self.folder_icon = QLabel()
        self.folder_icon.setPixmap(QPixmap("assets/icons/folder_closed.png"))
        self.folder_icon.setFixedSize(48, 48)
        self.folder_icon.setScaledContents(True)
        self.folder_icon.setCursor(Qt.PointingHandCursor)

        # Enable hover tracking
        self.folder_icon.setAttribute(Qt.WA_Hover, True)
        self.folder_icon.installEventFilter(self)

        # Add it to your layout (e.g., top-left corner or any layout you want)
        layout.addWidget(self.folder_icon)


        # Title
        pixel_font_title = QFont("Press Start 2P", 12)
        pixel_font_title.setStyleStrategy(QFont.NoAntialias)

        title = QLabel("Live Posture Monitoring")
        title.setFont(pixel_font_title)  # ✅ Now uses the pixel font
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: black;
            ont-weight: normal;
            padding: 10px;
        """)

        layout.addWidget(title)


        # Camera feed
        pixel_font = QFont("Press Start 2P", 8)
        pixel_font.setStyleStrategy(QFont.NoAntialias)

        self.image_label = QLabel("Click 'Start Camera' to begin webcam feed")
        self.image_label.setFont(pixel_font)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: transparent; border: none;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        # Status displays
        status_layout = QVBoxLayout()

        # Main posture status - big and prominent
        pixel_font = QFont("Press Start 2P", 10)
        pixel_font.setStyleStrategy(QFont.NoAntialias)

        self.posture_status = QLabel("Posture Status: Not monitoring")
        self.posture_status.setFont(pixel_font)
        self.posture_status.setAlignment(Qt.AlignCenter)
        self.posture_status.setStyleSheet("""
            font-size: 10px;  
            padding: 10px; 
            background-color: #fffff; 
            border: 3px solid #000000;
            border-bottom: 4px solid #000000;
            border-right: 4px solid #000000;
            color: black;             
        """)
        status_layout.addWidget(self.posture_status)

        pixel_font = QFont("Press Start 2P", 8)
        pixel_font.setStyleStrategy(QFont.NoAntialias)
        
        # Detailed stats display - smaller, secondary info
        self.stats_display = QLabel("Detailed Status: Click 'Start Camera' to begin monitoring")
        self.stats_display.setFont(pixel_font)
        self.stats_display.setAlignment(Qt.AlignCenter)
        self.stats_display.setStyleSheet("font-size: 10px; padding: 8px; background-color: #e8f4fd; border-radius: 6px; color: black;border-bottom: 4px solid #000000;  border-right: 4px solid #000000; border: 3px solid #000000;")
        status_layout.addWidget(self.stats_display)

        layout.addLayout(status_layout)

        # Control buttons
        btn_layout = QHBoxLayout()

        button_style = """
            QPushButton {
                    font-size: 10px;
                    padding: 6x 20px;
                    background-color: #f0f0f0;
                    border: 3px solid #000000;
                    border-bottom: 4px solid #000000;
                    border-right: 4px solid #000000;
                    border-radius: 10px;
                    color: #000;
                    min-width: 160px;
                    min-height: 32px;
                    font-family: "Press Start 2P", monospace;
                    font-size: 10px;

                }
            QPushButton:hover {
                background-color: #ccffcc;
        }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """
        icon_size = QSize(20, 20)

        # Start Camera Button
        self.start_button = QPushButton("  Start Camera")
        self.start_button.setIcon(QIcon("assets/start_icon.png"))
        self.start_button.setIconSize(icon_size)
        self.start_button.setFont(pixel_font)
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_video)
        btn_layout.addWidget(self.start_button)

        # Stop Camera Button 
        self.stop_button = QPushButton("  Stop Camera")
        self.stop_button.setIcon(QIcon("assets/stop_icon.png"))
        self.stop_button.setIconSize(icon_size)
        self.stop_button.setFont(pixel_font)
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)

        # Calibration Button
        self.calibrate_button = QPushButton("  Calibrate")
        self.calibrate_button.setIcon(QIcon("assets/calibrate_icon.png"))
        self.calibrate_button.setIconSize(icon_size)
        self.calibrate_button.setFont(pixel_font)
        self.calibrate_button.setStyleSheet(button_style + "background-color: #4CAF50; color: white;")
        self.calibrate_button.clicked.connect(self.start_calibration)
        btn_layout.addWidget(self.calibrate_button)


        
        
        layout.addLayout(btn_layout)

        # Voice command status

        pixel_font = QFont("Press Start 2P", 8)
        pixel_font.setStyleStrategy(QFont.NoAntialias)


        self.voice_status = QLabel("🎤 Voice Status: Initializing...")
        self.voice_status.setFont(pixel_font)
        self.voice_status.setAlignment(Qt.AlignCenter)
        self.voice_status.setStyleSheet("font-size: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px; color: black;")
        layout.addWidget(self.voice_status)

        self.live_tab.setLayout(QVBoxLayout())
        self.live_tab.layout().addWidget(live_wrapper)


    def init_log_tab(self):
        layout = QVBoxLayout()

        title = QLabel("Posture Data Log")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        #EMDYA CHANGE - Created a table widget for a more aesthetically pleasing log display
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make it read-only
        layout.addWidget(self.log_table)

        #Buttons for log management
        btn_layout = QHBoxLayout()
    
        load_button = QPushButton("📊 Load Posture Log")
        load_button.clicked.connect(self.load_log)
        btn_layout.addWidget(load_button)

        refresh_button = QPushButton("🔄 Refresh")
        refresh_button.clicked.connect(self.load_log)
        btn_layout.addWidget(refresh_button)

        layout.addLayout(btn_layout)
        self.log_tab.setLayout(layout)

        

        

         # Set layout to tab
        self.log_tab.setLayout(layout)
        

    def init_settings_tab(self):
        pixel_font = QFont("Press Start 2P", 10)
        pixel_font.setStyleStrategy(QFont.NoAntialias)
        


        layout = QVBoxLayout()

          #visual settings group
        visual_group = QGroupBox("Visual Settings")
        visual_group.setFont(pixel_font)  # Apply your pixel-style font
        visual_group.setStyleSheet("""
            QGroupBox {
                font-family: "Press Start 2P";
                font-size: 10px;
                color: black;
                border: 2px solid black;
                border-radius: 5px;
                margin-top: 10px;
                background-color: rgba(200, 200, 200, 160);  /* Transparent light gray */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        visual_layout = QFormLayout()
        # Pixel font
        pixel_font = QFont("Press Start 2P", 10)
        pixel_font.setStyleStrategy(QFont.NoAntialias)
        #landmark toggle
        self.landmark_checkbox = QCheckBox("Show pose landmarks on camera feed")
        self.landmark_checkbox.setFont(pixel_font)
        self.landmark_checkbox.setChecked(self.show_landmarks)
        self.landmark_checkbox.stateChanged.connect(self.toggle_landmark_visibility)

        landmark_label = QLabel("Landmarks:")
        landmark_label.setFont(pixel_font)
        visual_layout.addRow(landmark_label, self.landmark_checkbox)


         # info about the landmark toggle
        landmark_info = QLabel("When enabled, shows pose detection points and connections on the video feed")
        landmark_info.setFont(pixel_font)
        landmark_info.setStyleSheet("""
            font-size: 10px;
            color: #333;
            font-style: italic;
            font-family: "Press Start 2P";
        """)
        visual_layout.addRow("", landmark_info)

        visual_group.setLayout(visual_layout)
        layout.addWidget(visual_group)

        #notif_group = QGroupBox("Notification Settings")
        notif_group = QGroupBox()
        notif_group.setTitle("Notification Settings")
        notif_group.setFont(pixel_font)
        notif_group.setStyleSheet("""
            QGroupBox {
                font-family: "Press Start 2P";
                font-size: 10px;
                color: black;
                border: 2px solid black;
                border-radius: 5px;
                margin-top: 10px;
                background-color: rgba(200, 200, 200, 160);  /* Light gray with transparency */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        self.settings_tab.setStyleSheet("background-color: transparent;")


        notif_layout = QFormLayout()

        # Volume Control
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(self.notification_volume)
        self.volume_slider.setTickPosition(QSlider.TicksBelow)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

        self.volume_label = QLabel(f"{self.notification_volume}%")
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)
        notif_label1 = QLabel("Notification Volume:")
        notif_label1.setFont(pixel_font)
        notif_layout.addRow(notif_label1, volume_layout)
        self.volume_label.setFont(pixel_font)
        self.volume_label.setStyleSheet(""" 
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            background-color: white;
            border: 2px solid black;
            padding: 4px;
        """)

        # Beep Interval Control
        self.beep_interval_spinbox = QDoubleSpinBox()
        self.beep_interval_spinbox.setRange(0.5, 10.0)
        self.beep_interval_spinbox.setSingleStep(0.5)
        self.beep_interval_spinbox.setValue(self.beep_interval)
        self.beep_interval_spinbox.setSuffix(" seconds")
        self.beep_interval_spinbox.valueChanged.connect(self._on_beep_interval_changed)
        notif_label2 = QLabel("Beep Interval:")
        notif_label2.setFont(pixel_font)
        notif_layout.addRow(notif_label2, self.beep_interval_spinbox)
        # Apply pixel font and black styling to beep interval spinbox
        self.beep_interval_spinbox.setFont(pixel_font)
        self.beep_interval_spinbox.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            ackground-color: white;
            border: 2px solid black;
            padding: 4px;
        """)

        # Alert Duration Control
        self.alert_duration_spinbox = QSpinBox()
        self.alert_duration_spinbox.setRange(1, 60)
        self.alert_duration_spinbox.setValue(int(self.alert_duration))
        self.alert_duration_spinbox.setSuffix(" seconds")
        self.alert_duration_spinbox.valueChanged.connect(self._on_alert_duration_changed)
        notif_label3 = QLabel("Alert Duration:")
        notif_label3.setFont(pixel_font)
        notif_layout.addRow(notif_label3, self.alert_duration_spinbox)
        notif_group.setLayout(notif_layout)
        layout.addWidget(notif_group)
        # Apply pixel font and black styling to alert duration spinbox
        self.alert_duration_spinbox.setFont(pixel_font)
        self.alert_duration_spinbox.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            background-color: white;
            border: 2px solid black;
            padding: 4px;
        """)


        # Add some spacing
        layout.addStretch()

        # Voice Control Section
        #  Section Label (outside the box)
        voice_section_label = QLabel("🎤 Voice Control Settings")
        voice_section_label.setFont(pixel_font)
        voice_section_label.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            margin-top: 10px;
            margin-bottom: 2px;
            padding-left: 10px;
        """)
        layout.addWidget(voice_section_label)

        # 📦 Container Box for Voice Controls (no title)
        voice_group = QGroupBox()
        voice_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid black;
                border-radius: 5px;
                background-color: rgba(200, 200, 200, 160);
                margin-top: 0px;
            }
        """)
        voice_layout = QVBoxLayout()

        #  Checkbox (with pixel font)
        self.voice_checkbox = QCheckBox("Enable Voice Commands")
        self.voice_checkbox.setFont(pixel_font)
        self.voice_checkbox.setChecked(False)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice_recognition)
        self.voice_checkbox.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            padding: 5px;
        """)
        voice_layout.addWidget(self.voice_checkbox)

        #  Voice Commands Info Block
        voice_help_group = QGroupBox("Voice Commands Available")
        voice_help_group.setFont(pixel_font)
        voice_help_group.setStyleSheet("""
            QGroupBox {
                font-family: "Press Start 2P";
                font-size: 10px;
                color: black;
                border: 2px solid black;
                border-radius: 6px;
                background-color: rgba(200, 200, 200, 160);
            }
            QGroupBox::title {
                subcontrol-origin: content;
                subcontrol-position: top left;
                left: 6px;
                top: -2px;
                padding: 0px 4px;
            }
        """)

        voice_help_label = QLabel("""
        Camera Control:
        • "stop" only turns off camera – you can say "start" to turn it back on
        • App keeps running in background when camera is stopped
        • Only "exit" will close the entire application

        Speech Tips:
        • Use "stop" to pause camera, "exit" to close app
        • "cal" works better than "calibrate" for speech recognition
        • Wait for "Listening..." before speaking
        • Speak clearly at normal volume

        Recognition Examples:
        • "stop" → Camera off, app running
        • "start" → Camera on
        • "exit" → Close everything

        Note: Enable voice commands with the checkbox above first.
        """)
        voice_help_label.setFont(pixel_font)
        voice_help_label.setWordWrap(True)
        voice_help_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        voice_help_label.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            background-color: transparent;
            padding: 10px;
        """)

        voice_help_layout = QVBoxLayout()
        voice_help_layout.addWidget(voice_help_label)
        voice_help_group.setLayout(voice_help_layout)

        voice_layout.addWidget(voice_help_group)
        voice_group.setLayout(voice_layout)

        # Add the voice section to main layout
        layout.addWidget(voice_group)

            

        layout.addWidget(QLabel(""))  # Spacer

        # Data Management Section (Pixel Aesthetic)
        data_section = QLabel("📁 Data Management")
        data_section.setFont(pixel_font)
        data_section.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 12px;
            color: black;
            padding: 8px;
            background-color: rgba(255, 255, 255, 200);
            border: 2px solid black;
            border-radius: 10px;
            margin-top: 20px;
        """)
        layout.addWidget(data_section)
        data_btn_layout = QHBoxLayout()
        

        export_button = QPushButton("💾 Export Log as CSV")
        export_button.clicked.connect(self.export_log)
        data_btn_layout.addWidget(export_button)
        export_button.setFont(pixel_font)
        export_button.setStyleSheet("""
            QPushButton {
                font-family: "Press Start 2P";
                font-size: 10px;
                color: black;
                background-color: white;
                border: 2px solid black;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ccffcc;
            }
         """)

        clear_log_button = QPushButton("🗑️ Clear Log Data")
        clear_log_button.clicked.connect(self.clear_log)
        clear_log_button.setFont(pixel_font)
        clear_log_button.setStyleSheet("""
            QPushButton {
                font-family: "Press Start 2P";
                font-size: 10px;
                color: black;
                background-color: white;
                border: 2px solid black;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ffcccc;
            }
         """)
        clear_log_button.clicked.connect(self.clear_log)
        data_btn_layout.addWidget(clear_log_button)


        layout.addLayout(data_btn_layout)

        # Instructions
        info_label = QLabel("""
📋 Instructions:
1. Enable voice commands using the checkbox above
2. Start the camera feed in the 'Live Posture' tab  
3. Say "calibrate" or click the button to set your baseline posture
4. Maintain good posture during the 8-second calibration
5. The system will monitor your posture and provide audio alerts
6. View logged data in the 'Posture Log' tab

💡 Tips for Best Results:
• Ensure good lighting and a quiet environment
• Keep microphone permissions enabled for your browser/system
• Speak clearly when using voice commands
• Use a stable camera position (tripod recommended)
        """)
        info_label.setFont(pixel_font)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            font-family: "Press Start 2P";
            font-size: 10px;
            color: black;
            background-color: white;
            border: 2px solid black;
            border-radius: 10px;
            padding: 16px;
            margin-top: 12px;
        """)
        # Scrollable wrapper
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFixedHeight(200)
        info_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                width: 12px;
                background: rgba(220, 220, 220, 100);
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 6px;
            }
         """)
        info_scroll.setWidget(info_label)
        layout.addWidget(info_scroll)
        #layout.addWidget(info_label)

        layout.addStretch()
        # Wrapping full settings layout inside a container widget
        settings_inner_widget = QWidget()
        settings_inner_widget.setLayout(layout)
        # Scroll Wrapper for tabbed layout
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings_inner_widget)
        settings_scroll.setStyleSheet("""
            QScrollArea {
                    border: none;
                    background-color: transparent;
            }
            QScrollBar:vertical {
                    width: 12px;
                    background: rgba(220, 220, 220, 100);
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 6px;
            }
         """)
        #  Set main layout of the tab
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(settings_scroll)
        self.settings_tab.setLayout(outer_layout)
        

    def toggle_landmark_visibility(self, state):
        self.show_landmarks = (state == Qt.Checked)
        status = "enabled" if self.show_landmarks else "disabled"
        print(f"[GUI] Landmark visibility {status}")

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_landmark_visibility(self.show_landmarks)
        # Provide user feedback
        if self.show_landmarks:
            self.stats_display.setText("Landmarks enabled - pose detection points will be visible")
        else:
            self.stats_display.setText("Landmarks disabled - clean video feed") 

    def _on_volume_changed(self, value):
        self.notification_volume = value
        self.volume_label.setText(f"{value}%")
        backend.update_notification_volume(value)

    def _on_beep_interval_changed(self, value):
        self.beep_interval = value
        backend.update_beep_interval(value)

    def _on_alert_duration_changed(self, value):
        self.alert_duration = value
        backend.update_alert_duration(value)

    def toggle_voice_recognition(self, state):
        """Enable or disable voice recognition"""
        if state == Qt.Checked:
            self.speech_thread.enable_listening()
            self.voice_status.setText("🎤 Voice Status: Listening for commands...")
            self.voice_status.setStyleSheet(
                "font-size: 12px; padding: 5px; background-color: #d4edda; border-radius: 3px;")
            print("[GUI] Voice recognition enabled")
        else:
            self.speech_thread.disable_listening()
            self.voice_status.setText("🎤 Voice Status: Disabled")
            self.voice_status.setStyleSheet(
                "font-size: 12px; padding: 5px; background-color: #f8d7da; border-radius: 3px;")
            print("[GUI] Voice recognition disabled")

    def handle_voice_command(self, command):
        """Handle recognized voice commands with improved matching"""
        print(f"[GUI] Processing voice command: '{command}'")

        # Convert to lowercase and split into words for better matching
        words = command.lower().split()
        command_lower = command.lower()

        # DEBUG: Show what we're analyzing
        print(f"[DEBUG] Words: {words}")
        print(f"[DEBUG] Command lower: '{command_lower}'")

        # Show what was heard in the GUI
        self.voice_status.setText(f"🎤 Heard: '{command}'")

        # CALIBRATION COMMANDS - More flexible matching for "calibrate"
        calibration_triggers = [
            "calibrate", "calibration", "collab", "cal", "caliber",
            "collaborate", "calib", "kelly", "cali", "cab", "start calibration"
        ]
        if any(trigger in command_lower for trigger in calibration_triggers):
            print("[VOICE] ✅ Calibration command detected")
            self.start_calibration()
            self.stats_display.setText("🎤 Voice Command: Starting calibration...")

        # STOP CAMERA COMMANDS - Only stops camera, keeps app running (CHECK THIS FIRST)
        elif any(phrase in command_lower for phrase in
                 ["stop camera", "pause camera", "turn off camera", "camera off"]) or (
                len(words) == 1 and words[0] in ["stop", "pause", "halt", "off"]):
            print("[VOICE] ✅ Stop camera command detected (app stays open)")
            if self.video_thread.isRunning():
                self.stop_video()
                self.stats_display.setText("🎤 Camera stopped - app still running. Say 'start' to resume.")
            else:
                self.stats_display.setText("🎤 Camera already stopped. Say 'start' to begin.")

        # EXIT COMMANDS - Only for closing the entire app (MORE SPECIFIC)
        elif any(phrase in command_lower for phrase in
                 ["exit", "quit", "close app", "goodbye", "end app", "close application", "shut down"]):
            print("[VOICE] ✅ Exit application command detected (closing app)")
            self.stats_display.setText("🎤 Voice Command: Exiting application...")
            QTimer.singleShot(1000, self.close)  # Close after 1 second

        # START COMMANDS (more flexible matching)
        elif any(word in ["start", "begin", "go", "play", "run", "on"] for word in words) or any(
                phrase in command_lower for phrase in ["turn on", "start camera", "begin camera"]):
            print("[VOICE] ✅ Start camera command detected")
            if not self.video_thread.isRunning():
                self.start_video()
                self.stats_display.setText("🎤 Camera started successfully!")
            else:
                self.stats_display.setText("🎤 Camera already running")

        # HELP COMMAND
        elif any(word in ["help", "commands", "what", "options"] for word in words):
            print("[VOICE] Help command detected")
            self.stats_display.setText("🎤 Commands: 'start' (camera), 'stop' (camera), 'cal' (calibrate), 'exit' (app)")

        else:
            print(f"[VOICE] ❌ Unknown command: '{command}'")
            print(f"[VOICE] Commands: stop (camera only), exit (close app), start (camera), cal (calibrate)")
            self.stats_display.setText(f"🎤 Unknown: '{command}' - Try: stop, start, cal, exit")

        # Reset voice status after 4 seconds to show the feedback longer
        QTimer.singleShot(4000, lambda: self.reset_voice_status())

    def update_speech_status(self, status):
        """Update speech recognition status"""
        if self.voice_checkbox.isChecked():
            self.voice_status.setText(f"🎤 Voice Status: {status}")

    def reset_voice_status(self):
        """Reset voice status to default listening state"""
        if self.voice_checkbox.isChecked():
            self.voice_status.setText("🎤 Voice Status: Listening for commands...")
            self.voice_status.setStyleSheet(
                "font-size: 12px; padding: 5px; background-color: #d4edda; border-radius: 3px;")

    def start_calibration(self):
        """Start the calibration process"""
        import backend
        backend.calibration_start_time = time.time()
        backend.is_calibrating = True
        backend.calibration_data = {k: [] for k in backend.calibration_data}
        print("[GUI] Calibration started")
        self.stats_display.setText("⚙️ Calibration started! Maintain good posture for 8 seconds...")
    def load_log(self):
        
        try:
            log_path = os.path.join(os.path.dirname(__file__), "posture_log_export.csv")

            print(f"[DEBUG] Absolute path: {os.path.abspath(log_path)}")
            print(f"[DEBUG] File exists: {os.path.exists(log_path)}")

            if os.path.exists(log_path):
                expected_columns = ["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"]
                df = pd.read_csv(log_path, header=None, names=expected_columns)

                
                

                # Show only recent entries
                display_df = df[expected_columns].tail(50).reset_index()

                self.log_table.setColumnCount(len(expected_columns))
                self.log_table.setHorizontalHeaderLabels(expected_columns)
                self.log_table.setRowCount(len(display_df))

                for row_idx, row in display_df.iterrows():
                    for col_idx, col_name in enumerate(expected_columns):
                        value = row[col_name]
                        if pd.isna(value):
                            value = "—"
                        item = QTableWidgetItem(str(value))

                        # ✨ Default to white text
                        item.setForeground(Qt.white)

                        # 🎨 Highlight Posture Status
                        if col_name == "Posture Status":
                            label = str(value).lower()
                            if "bad" in label:
                                item.setForeground(Qt.red)
                            elif "good" in label:
                                item.setForeground(Qt.green)
                            elif "moderate" in label:
                                item.setForeground(Qt.darkYellow)

                        self.log_table.setItem(row_idx, col_idx, item)

                # Optional: enable dark theme styling
                self.log_table.setStyleSheet("""
                    QTableWidget {
                        background-color: #1e1e1e;
                        color: white;
                        gridline-color: #444;
                    }
                    QHeaderView::section {
                        background-color: #2d2d2d;
                        color: white;
                        font-weight: bold;
                    }
                """)

            else:
                self.log_table.setRowCount(1)
                self.log_table.setColumnCount(1)
                self.log_table.setItem(0, 0, QTableWidgetItem("📂 No log file found."))

        except Exception as e:
            self.log_table.setRowCount(1)
            self.log_table.setColumnCount(1)
            self.log_table.setItem(0, 0, QTableWidgetItem(f"❌ Error loading log: {e}"))

    def clear_log(self):
        try:
            if os.path.exists("posture_trend_log.csv"):
                os.remove("posture_trend_log.csv")
                print("[INFO] Log file cleared")
                self.log_table.setRowCount(1)
                self.log_table.setItem(0, 0, QTableWidgetItem("🗑️ Log data cleared."))
            else:
                self.log_table.setRowCount(1)
                self.log_table.setItem(0, 0, QTableWidgetItem("📂 No log file to clear."))
        except Exception as e:
            print(f"[ERROR] Failed to clear log: {e}")

    def export_log(self):
        if os.path.exists("posture_trend_log.csv"):
            dest, _ = QFileDialog.getSaveFileName(
                self,
                "Save Log",
                "posture_log_export.csv",
                "CSV Files (*.csv)"
            )
            if dest:
                try:
                    with open("posture_trend_log.csv", "r") as src:
                        with open(dest, "w") as dst:
                            dst.write(src.read())
                    print(f"[INFO] Log exported to {dest}")
                    self.stats_display.setText(f"💾 Log exported successfully!")
                except Exception as e:
                    print(f"[ERROR] Failed to export: {e}")
                    self.stats_display.setText(f"❌ Export failed: {e}")
        else:
            self.stats_display.setText("📂 No log file found to export")

    def start_video(self):
        print("[INFO] Start button clicked")

        # Stop any existing thread first
        if self.video_thread and self.video_thread.isRunning():
            print("[INFO] Stopping existing video thread...")
            self.video_thread._run_flag = False
            self.video_thread.wait(1000)  # Wait up to 1 second

        # Create a completely new video thread
        print("[INFO] Creating new video thread...")
        self.video_thread = VideoThread(show_landmarks=self.show_landmarks)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_stats_signal.connect(self.update_stats)



        # Start the new thread
        self.video_thread.start()

        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stats_display.setText(" Camera starting...")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            background-color: transparent;
            border: none;
        """)
        self.image_label.setText("")  # Remove placeholder
        
        # Set initial monitoring state - this won't change until we get stable results
        self.posture_status.setText("Monitoring Posture...")
        self.posture_status.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            padding: 15px; 
            background-color: #cce5ff; 
            border: 2px solid #007bff;
            border-radius: 10px;
            color: #004085;
        """)

        print("[INFO] New video thread started")

    def stop_video(self):
        print("[INFO] Stop button clicked - stopping camera only, keeping app open")

        if self.video_thread and self.video_thread.isRunning():
            print("[INFO] Video thread is running, stopping it...")

            # Set the stop flag first
            self.video_thread._run_flag = False
            print("[INFO] Set video thread stop flag")

            # Give the thread a moment to finish its current iteration
            QTimer.singleShot(100, self.finish_video_stop)
        else:
            print("[INFO] Video thread was not running")
            self.finish_video_stop()

    def finish_video_stop(self):
        """Complete the video stopping process"""
        print("[INFO] Finishing video stop process...")

        # Don't call wait() - just let the thread finish naturally
        if self.video_thread and self.video_thread.isRunning():
            print("[INFO] Thread still running, waiting a bit more...")
            # Give it more time and try again
            QTimer.singleShot(500, self.force_video_stop)
        else:
            print("[INFO] Thread stopped naturally")
            self.update_ui_after_stop()

    def force_video_stop(self):
        """Force stop if thread doesn't stop naturally"""
        print("[INFO] Force stopping video thread...")

        if self.video_thread and self.video_thread.isRunning():
            try:
                # Try terminate as last resort, but don't wait indefinitely
                self.video_thread.terminate()
                print("[INFO] Thread terminated")
            except Exception as e:
                print(f"[INFO] Error terminating thread: {e}")

        self.update_ui_after_stop()

    def update_ui_after_stop(self):
        """Update UI after video is stopped"""
        print("[INFO] Updating UI after video stop...")

        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stats_display.setText(" Camera stopped")
        self.posture_status.setText(" Camera Stopped")
        self.posture_status.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            padding: 15px; 
            background-color: #f0f0f0; 
            border: 2px solid #ccc;
            border-radius: 10px;
            color: #333;
        """)
        pixel_font = QFont("Press Start 2P", 8)
        pixel_font.setStyleStrategy(QFont.NoAntialias)
        self.image_label.setFont(pixel_font)
        self.image_label.setText("Click 'Start Camera' to begin webcam feed")
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText("Click 'Start Camera' to begin webcam feed")
        self.image_label.clear()  # Clear any existing image

        print("[INFO] UI updated - app should remain open")

        # Verify app is still alive
        QTimer.singleShot(1000, self.check_app_status)

    def check_app_status(self):
        """Check if the app is still running after stop command"""
        print("[DEBUG] App status check - if you see this, the app is still running!")
        self.stats_display.setText("⏹️ Camera stopped - App is running normally")
    
    def eventFilter(self, source, event):
        if source == self.folder_icon:
            if event.type() == QEvent.Enter:
                self.folder_icon.setPixmap(QPixmap("assets/icons/folder_open.png"))
            elif event.type() == QEvent.Leave:
                self.folder_icon.setPixmap(QPixmap("assets/icons/folder_closed.png"))
        return super().eventFilter(source, event)


    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def update_stats(self, text):
        # Always update the detailed analysis section immediately
        current_text = self.stats_display.text()
        if "🎤 Voice Command:" not in current_text:
            self.stats_display.setText(f"Analysis: {text}")
        
        # Only update the main posture status for STABLE, confirmed results
        # Don't update for transitioning states or temporary detections
        if text and not any(unstable_word in text.lower() for unstable_word in
                            ["detecting", "stabilizing", "transitioning", "confirming", "analyzing"]):

            # Only update main status for confident, stable results
            if "good posture" in text.lower():
                self.posture_status.setText(" Good Posture")
                self.posture_status.setStyleSheet("""
                    font-size: 20px; 
                    font-weight: bold; 
                    padding: 15px; 
                    background-color: #d4edda; 
                    border: 2px solid #28a745;
                    border-radius: 10px;
                    color: #155724;
                """)
            elif "moderately bad posture" in text.lower() or "moderate" in text.lower():
                self.posture_status.setText("Moderate Posture Issues")
                self.posture_status.setStyleSheet("""
                    font-size: 20px; 
                    font-weight: bold; 
                    padding: 15px; 
                    background-color: #fff3cd; 
                    border: 2px solid #ffc107;
                    border-radius: 10px;
                    color: #856404;
                """)
            elif "bad posture" in text.lower():
                self.posture_status.setText(" Poor Posture Detected")
                self.posture_status.setStyleSheet("""
                    font-size: 20px; 
                    font-weight: bold; 
                    padding: 15px; 
                    background-color: #f8d7da; 
                    border: 2px solid #dc3545;
                    border-radius: 10px;
                    color: #721c24;
                """)
            elif "no pose" in text.lower():
                self.posture_status.setText(" No Person Detected")
                self.posture_status.setStyleSheet("""
                    font-size: 20px; 
                    font-weight: bold; 
                    padding: 15px; 
                    background-color: #e2e3e5; 
                    border: 2px solid #6c757d;
                    border-radius: 10px;
                    color: #495057;
                """)

        # For transitioning/analyzing states, keep current main status but show activity in detailed section
        # The main status will only change when we get a confirmed, stable result

    def closeEvent(self, event):
        """Handle window closing"""
        print("[GUI]  closeEvent triggered - checking why...")
        
        # Print stack trace to see what's calling close
        import traceback
        print("[GUI] Close event stack trace:")
        traceback.print_stack()

        print("[GUI] Stopping threads before closing...")

        # Stop video thread
        if self.video_thread.isRunning():
            print("[GUI] Stopping video thread...")
            self.video_thread.stop()

        # Stop speech thread
        if self.speech_thread.isRunning():
            print("[GUI] Stopping speech thread...")
            self.speech_thread.stop()

        print("[GUI] All threads stopped, accepting close event")
        event.accept()
        print("[GUI] Application closed")


# Main application launcher
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
