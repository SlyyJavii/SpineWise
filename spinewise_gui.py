import os, queue, cv2, csv, time, numpy as np, pandas as pd, mediapipe as mp, threading, backend, speech_recognition as sr
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QStackedWidget, QButtonGroup, QRadioButton, QSizePolicy, QFrame,
    QVBoxLayout, QWidget, QTabWidget, QMainWindow, QFileDialog, QTextEdit, QDoubleSpinBox,
    QScrollArea, QSpinBox, QHBoxLayout, QCheckBox, QFormLayout, QSlider, QGroupBox, QProgressBar,
    QTableWidgetItem, QTableWidget, QGridLayout, QHeaderView, QAction, QLineEdit, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QImage, QDesktopServices, QPixmap, QFont, QIcon, QFontDatabase, QPalette, QBrush, QPainter, QColor
from PyQt5.QtCore import Qt, QUrl, QSize, QPropertyAnimation, QRect, QEasingCurve, QThread, pyqtSignal, QEvent, QTimer, \
    QObject, QPoint

from backend import (
    analyze_posture, get_pose_landmarker, get_face_landmarker, draw_landmarks,
    normalize_lighting, is_calibrating, calibration_start_time, calibration_data,
    set_gui_mode
)

# theme importing
from spinewise_theme import (
    apply_palette, APP_QSS, RECS_QSS, PRODUCT_CARD_QSS, LIVE_IMAGE_QSS, STATUS_PANEL_QSS,
    BTN_SUCCESS_QSS, POPUP_FRAME_QSS, DOT_QSS, posture_style, voice_status_style
)
from notification_widget import (
    NotificationButton, NotificationDropdown,
    NotificationItem, NotificationBadge
)
from voice_config import voice_config
from datetime import datetime
from poll import reader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class GraphThread(QThread):  # Tried changing from QObject to QThread as a dummy fix
    # Seems to have helped here and there with minimizing crashes (It also now works on my machine?)
    # Also, considering that having poll.py was not all that necessary due to its small size,
    # did a dummy overhaul that should work much the same
    # Honestly it's not even really an overhaul. Jason did great work initially, I just wanted to
    # fix it so it wouldn't fight
    # This is honestly going to be my focus for next sprint
    # Because notification system sucked the life out of me.
    finished = pyqtSignal()
    progress = pyqtSignal()
    update_plot = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.file_path = "posture_trend_log.csv"
        self._is_running = False
        self.tab_active = False
        self.file_position = 0

        plt.style.use('bmh')
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.ax1 = self.figure.add_subplot(121)
        self.canvas.ax2 = self.figure.add_subplot(122)
        self.xdata = [0]
        self.ydata = [[0], [0]]

    def set_tab(self, index):
        self.tab_active = (index == 1)

    def run(self):
        self._is_running = True
        while self._is_running:
            if self.tab_active:
                self.progress.emit()
                try:
                    self.read_new_data_incrementally()
                    self.update_plot.emit()
                except Exception as e:
                    print(f"[ANALYTICS] Error updating plot: {e}")

            self.msleep(1000)

        self.finished.emit()

    def read_new_data_incrementally(self):
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, 'r') as file:
                file.seek(self.file_position)

                csv_reader = csv.reader(file)
                interval = 3

                for row in csv_reader:
                    if len(row) >= 6 and row[-1].replace('.', '').replace('-', '').isdigit():
                        try:
                            confidence = int(float(row[-1]))
                            head_tilt = float(row[4])
                            self.ydata[0].append(confidence)
                            self.ydata[1].append(head_tilt)
                            self.xdata.append(self.xdata[-1] + interval)
                        except (ValueError, IndexError):
                            continue

                self.file_position = file.tell()

        except Exception as e:
            print(f"[ANALYTICS] Failed to read CSV: {e}")

    def plot_on_main_thread(self):
        try:
            self.canvas.ax1.cla()
            self.canvas.ax1.plot(self.xdata, self.ydata[0], marker='o', linewidth=1)
            self.canvas.ax1.set_title("Confidence (0-7 scaled) over time")
            self.canvas.ax1.set_xlabel("Sample")
            self.canvas.ax1.set_ylabel("Confidence (0-7 scaled)")
            self.canvas.ax1.grid(True)

            self.canvas.ax2.cla()
            self.canvas.ax2.plot(self.xdata, self.ydata[1], marker='o', linewidth=1)
            self.canvas.ax2.set_title("Head tilt (normalized) over time")
            self.canvas.ax2.set_xlabel("Sample")
            self.canvas.ax2.set_ylabel("Head tilt (normalized)")
            self.canvas.ax2.grid(True)

            self.figure.tight_layout()
            self.canvas.draw()
            # Thankfully has stopped happening to me
            # This used to keep occurring with the old version
            # It may be machine dependent. Not sure. Not like it matters right now.
        except Exception as e:
            print(f"[ANALYTICS] uh oh! plot drawing failed!: {e}")

    def stop(self):
        self._is_running = False
        self.wait(2000)

# speech thread
class SpeechRecognitionThread(QThread):
    command_detected = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.listening_enabled = False
        self.recognizer = None
        self.microphone = None
        self.show_landmarks = False

    def run(self):
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.recognizer.energy_threshold = 100
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5
            self.recognizer.phrase_threshold = 0.2
            self.recognizer.non_speaking_duration = 0.3
            self.status_update.emit("Calibrating microphone...")

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            if self.recognizer.energy_threshold > 300:
                self.recognizer.energy_threshold = 200

            self.status_update.emit("Ready - try saying 'start'")

            while self._run_flag:
                if not self.listening_enabled:
                    time.sleep(0.5)
                    continue

                try:
                    with self.microphone as source:
                        self.status_update.emit("Listening...")
                        audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)

                    self.status_update.emit("Processing...")

                    # Use configured language
                    current_language = voice_config.get_language()
                    command = None

                    try:
                        command = self.recognizer.recognize_google(audio, language=current_language).lower().strip()
                    except sr.UnknownValueError:
                        # Try fallback to English if not already English
                        if current_language != "en-US":
                            try:
                                command = self.recognizer.recognize_google(audio, language="en-US").lower().strip()
                            except:
                                continue
                        else:
                            continue

                    if command:
                        self.command_detected.emit(command)
                        self.status_update.emit(f"Heard: '{command}'")

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.status_update.emit("Could not understand")
                    continue
                except sr.RequestError:
                    self.status_update.emit("Speech API error")
                    time.sleep(5)
                    continue
                except Exception:
                    continue

        except Exception as e:
            self.status_update.emit(f"Mic init failed: {e}")

    def enable_listening(self): self.listening_enabled = True
    def disable_listening(self): self.listening_enabled = False

    def stop(self):
        self._run_flag = False
        self.listening_enabled = False
        if not self.wait(2000):  # Wait up to 2 seconds
            print("[WARNING] Speech thread went oopsie! Go tell Juan.")

# video thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_stats_signal = pyqtSignal(str)

    def __init__(self, show_landmarks=False):
        super().__init__()
        self._run_flag = True
        self.pose_landmarker = None
        self.face_landmarker = None
        self.raw_queue = None
        self.processed_queue = None
        self.show_landmarks = show_landmarks

    def set_landmark_visibility(self, show_landmarks): self.show_landmarks = show_landmarks

    def process_image_queue(self):
        with self.pose_landmarker as pose_landmarker, self.face_landmarker as face_landmarker:
            while self._run_flag:
                frame = self.raw_queue.get()
                frame = normalize_lighting(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = int(round(time.time() * 1000))
                pose_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                face_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                pose_results = pose_landmarker.detect_for_video(pose_image, timestamp)
                face_results = face_landmarker.detect_for_video(face_image, timestamp)
                annotated = np.copy(frame)
                if pose_results.pose_landmarks:
                    if self.show_landmarks:
                        draw_landmarks(annotated, pose_results.pose_landmarks)
                    result = analyze_posture(
                        annotated,
                        pose_results.pose_landmarks[0],
                        face_results.face_landmarks if face_results.face_landmarks else None
                    )
                    self.update_stats_signal.emit(result)
                else:
                    self.update_stats_signal.emit("No pose detected")
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.processed_queue.put(qt_image)

    def run(self):
        self.pose_landmarker = get_pose_landmarker()
        self.face_landmarker = get_face_landmarker()
        self.raw_queue = queue.Queue()
        self.processed_queue = queue.Queue()
        threading.Thread(target=self.process_image_queue, daemon=True).start()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to 1

        try:
            while self._run_flag:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame")
                    continue

                if not self.raw_queue.full():
                    self.raw_queue.put(frame)

                try:
                    processed_frame = self.processed_queue.get(timeout=0.1)
                    self.change_pixmap_signal.emit(processed_frame)
                except:
                    pass

        finally:
            cap.release()
            cv2.destroyAllWindows()


    def stop(self):
        self._run_flag = False

# product card
class ProductCard(QFrame):
    def __init__(self, title, category, why, price_text, rating=None, reviews=None, url=None, parent=None):
        super().__init__(parent)
        self.setObjectName("ProductCard")
        self.setStyleSheet(PRODUCT_CARD_QSS)
        v = QVBoxLayout(self)
        title_lbl = QLabel(title or "—")
        title_lbl.setObjectName("CardTitle")
        title_lbl.setWordWrap(True)
        pill = QLabel(category or "—")
        pill.setObjectName("Pill")
        top = QHBoxLayout(); top.addWidget(title_lbl, 1); top.addWidget(pill, 0, Qt.AlignRight); v.addLayout(top)
        why_lbl = QLabel(why or "—"); why_lbl.setObjectName("Why"); why_lbl.setWordWrap(True); v.addWidget(why_lbl)
        meta = [];
        if rating: meta.append(f"⭐ {rating}")
        if reviews: meta.append(f"({reviews} reviews)")
        meta_lbl = QLabel(" ".join(meta) if meta else " "); meta_lbl.setObjectName("Meta"); v.addWidget(meta_lbl)
        bottom = QHBoxLayout()
        price_lbl = QLabel(price_text or "—"); price_lbl.setObjectName("Price"); bottom.addWidget(price_lbl)
        if url:
            link = QLabel(f'<a href="{url}">Open</a>'); link.setOpenExternalLinks(True); bottom.addStretch(1); bottom.addWidget(link)
        v.addLayout(bottom)

# main window
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        set_gui_mode(True)
        self.setWindowTitle("SpineWise Posture App - With Voice Control")
        self.setGeometry(100, 100, 1400, 900)

        apply_palette(self)
        self.setStyleSheet(APP_QSS)
        self.app_font = QFont("Segoe UI", 10)

        central = QWidget()
        main_container_layout = QVBoxLayout()
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(0)

        self.notification_btn = NotificationButton()
        self.notification_dropdown = NotificationDropdown()
        self.notification_dropdown.setParent(self)  # OK to keep parent set to the window
        self.notification_dropdown.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.notification_dropdown.setAttribute(Qt.WA_TranslucentBackground)
        self.notification_dropdown.setWindowModality(Qt.NonModal)
        self.notification_dropdown.hide()
        self.notification_btn.clicked.connect(self.toggle_notification_dropdown)

        self.last_posture_state = "good"
        self.bad_posture_start_time = None
        self.posture_notification_cooldown = 60  # seconds between notifications
        self.last_notification_time = None

        self.dropdown_animation = None
        self.toast_fade_in = None
        self.toast_fade_out = None

        self.notification_dropdown.raise_()

        toolbar_widget = self.create_toolbar()
        main_container_layout.addWidget(toolbar_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setElideMode(Qt.ElideRight)
        main_container_layout.addWidget(self.tab_widget, 1)  # Give tab widget stretch priority

        central.setLayout(main_container_layout)
        self.setCentralWidget(central)

        self.live_tab = QWidget()
        self.analytics_tab = QWidget()
        self.recommendations_tab = QWidget()
        self.settings_tab = QWidget()

        self.tab_widget.addTab(self.live_tab, "Live Posture")
        self.tab_widget.addTab(self.analytics_tab, "Analytics Tab")
        self.tab_widget.addTab(self.recommendations_tab, "Recommendations")
        self.tab_widget.addTab(self.settings_tab, "Settings")

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

        self.init_live_tab()
        self.init_analytics_tab()
        self.init_settings_tab()

        # Default engine mode
        # Sets GUI dropdown and backend mode to rule based at startup
        if hasattr(self, "engine_combo"):
            self.engine_combo.setCurrentIndex(0)
            backend.set_detection_mode("rules")

        self._active_animations = []

        self.init_recommendations_tab()
        self.speech_thread.start()

    def create_toolbar(self):
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(16, 8, 16, 8)

        title_label = QLabel("SpineWise Posture Monitoring")
        title_label.setStyleSheet("""
                    QLabel {
                        color: #0B5CAD;
                        font-size: 16px;
                        font-weight: bold;
                        font-family: 'Segoe UI', sans-serif;
                        margin-left: 8px;
                    }
                """)
        toolbar_layout.addWidget(title_label)

        toolbar_layout.addStretch()

        toolbar_layout.addSpacing(10)

        toolbar_layout.addWidget(self.notification_btn)

        toolbar_widget.setLayout(toolbar_layout)
        toolbar_widget.setFixedHeight(60)
        toolbar_widget.setStyleSheet("""
                    QWidget {
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                            stop: 0 #FFFFFF, stop: 1 #F8F9FA);
                        border-bottom: 1px solid #DEE2E6;
                    }
                """)

        return toolbar_widget

        #settings handler for engine switch
    def _on_engine_changed(self, idx):
        mode = self.engine_combo.itemData(idx)
        try:
            backend.set_detection_mode(mode)
            if hasattr(self, "stats_display"):
                self.stats_display.setText(f"Detection engine set to: {self.engine_combo.currentText()}")
        except Exception as e:
            if hasattr(self, "stats_display"):
                self.stats_display.setText(f"Failed to switch engine: {e}")

    def _on_refresh_products_from_csv(self):
        base = os.path.dirname(__file__)
        files = [
            ("Neck Pillow", os.path.join(base, "neck_pillow_results.csv")),
            ("Posture Corrector", os.path.join(base, "posture_corrector_results.csv")),
            ("Resistance Bands", os.path.join(base, "resistance_bands_results.csv")),
        ]
        products = []
        for category, path in files:
            if not os.path.exists(path): continue
            try:
                df = pd.read_csv(path)
                for _, row in df.head(2).iterrows():
                    products.append({
                        "title": str(row.get("Title", "—")),
                        "category": category,
                        "why": f"{category} can help improve posture by providing targeted support or training.",
                        "confidence": None,
                        "price_text": str(row.get("Print_Price", "—")),
                        "rating": str(row.get("Rating", "")) if not pd.isna(row.get("Rating", "")) else "",
                        "reviews": str(row.get("Reviews", "")) if not pd.isna(row.get("Reviews", "")) else "",
                        "url": str(row.get("Link", "")),
                    })
            except Exception as e:
                print("[RECS] CSV read failed:", path, e)
        if not products:
            products = [
                {"title": "Memory Foam Neck Pillow", "category": "Neck Pillow", "why": "Supports cervical alignment during long sessions.", "confidence": None, "price_text": "$24.99", "rating": "4.5", "reviews": "12,345", "url": ""},
                {"title": "Adjustable Posture Corrector", "category": "Posture Corrector", "why": "Gently retracts shoulders to counter slouching.", "confidence": None, "price_text": "$29.99", "rating": "4.3", "reviews": "8,901", "url": ""},
                {"title": "Resistance Bands Set", "category": "Resistance Bands", "why": "Helps strengthen scapular stabilizers.", "confidence": None, "price_text": "$19.99", "rating": "4.6", "reviews": "22,101", "url": ""},
            ]
        self._display_product_cards(products)
        self._populate_recs_table(products)

    def _display_product_cards(self, products):
        while hasattr(self, "product_grid") and self.product_grid.count():
            w = self.product_grid.takeAt(0).widget()
            if w: w.deleteLater()
        cols = 3
        for i, p in enumerate(products):
            r, c = divmod(i, cols)
            card = ProductCard(
                title=p.get("title"), category=p.get("category"), why=p.get("why"),
                price_text=p.get("price_text"), rating=p.get("rating"), reviews=p.get("reviews"), url=p.get("url")
            )
            self.product_grid.addWidget(card, r, c)
        self.product_grid.setRowStretch((len(products) + cols - 1) // cols + 1, 1)

    def init_recommendations_tab(self):
        self.recommendations_tab.setObjectName("RecsTab")
        root = QWidget(); root.setObjectName("RecsRoot"); root.setStyleSheet(RECS_QSS)
        outer = QVBoxLayout(root); outer.setContentsMargins(16,16,16,16); outer.setSpacing(14)

        cards_title = QLabel("Recommended Products"); cards_title.setProperty("class", "SectionTitle"); cards_title.setObjectName("SectionTitle"); cards_title.setAlignment(Qt.AlignLeft)
        outer.addWidget(cards_title)

        tools = QHBoxLayout()
        refresh_btn = QPushButton("Refresh from CSV"); refresh_btn.setObjectName("Primary"); refresh_btn.setProperty("class", "Primary"); refresh_btn.clicked.connect(self._on_refresh_products_from_csv)
        tools.addWidget(refresh_btn, 0); tools.addStretch(1); outer.addLayout(tools)

        self.products_container = QWidget()
        self.product_grid = QGridLayout(self.products_container)
        self.product_grid.setContentsMargins(0,0,0,0)
        self.product_grid.setHorizontalSpacing(12)
        self.product_grid.setVerticalSpacing(12)
        outer.addWidget(self.products_container)

        controls_title = QLabel("Fine-tune & Export"); controls_title.setObjectName("SubTitle"); controls_title.setProperty("class", "SubTitle")
        outer.addWidget(controls_title)

        filters_row = QHBoxLayout()
        issue_lbl = QLabel("Focus issues:"); self.issue_filter_input = QLineEdit(); self.issue_filter_input.setPlaceholderText("e.g., forward head, rounded shoulders")
        filters_row.addWidget(issue_lbl); filters_row.addWidget(self.issue_filter_input, 1)
        budget_lbl = QLabel("Budget:"); self.budget_input = QLineEdit(); self.budget_input.setPlaceholderText("Max $ (optional)"); self.budget_input.setFixedWidth(160)
        filters_row.addWidget(budget_lbl); filters_row.addWidget(self.budget_input)
        outer.addLayout(filters_row)

        self.recs_table = QTableWidget(); self.recs_table.setColumnCount(6)
        self.recs_table.setHorizontalHeaderLabels(["Product", "Category", "Why it helps", "Confidence", "Price", "Link"])
        self.recs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        outer.addWidget(self.recs_table)

        buttons = QHBoxLayout()
        generate_btn = QPushButton("Generate Recommendations"); generate_btn.setObjectName("Primary"); generate_btn.setProperty("class", "Primary"); generate_btn.clicked.connect(self._on_generate_recommendations)
        save_btn = QPushButton("Save as CSV"); save_btn.clicked.connect(self._on_save_recommendations_csv)
        buttons.addWidget(generate_btn); buttons.addWidget(save_btn); buttons.addStretch(1); outer.addLayout(buttons)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(root)
        main_layout = QVBoxLayout(); main_layout.addWidget(scroll); self.recommendations_tab.setLayout(main_layout)
        self._on_refresh_products_from_csv()

    # live tab
    def init_live_tab(self):
        live_wrapper = QWidget(); live_wrapper.setAttribute(Qt.WA_StyledBackground, True); live_wrapper.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(live_wrapper); layout.setSpacing(12)

        self.folder_icon = QLabel(); self.folder_icon.setPixmap(QPixmap("assets/icons/folder_closed.png")); self.folder_icon.setFixedSize(40,40)
        self.folder_icon.setScaledContents(True); self.folder_icon.setCursor(Qt.PointingHandCursor)
        self.folder_icon.setAttribute(Qt.WA_Hover, True); self.folder_icon.installEventFilter(self)
        top_row = QHBoxLayout(); top_row.addWidget(self.folder_icon, 0, Qt.AlignLeft); top_row.addStretch(1); layout.addLayout(top_row)

        title = QLabel("Live Posture Monitoring"); title.setFont(QFont(self.app_font.family(), 18, QFont.DemiBold)); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("color: #0B5CAD;")
        layout.addWidget(title)

        self.image_label = QLabel("Click 'Start Camera' to begin webcam feed"); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setMinimumSize(800, 480)
        self.image_label.setStyleSheet(LIVE_IMAGE_QSS); self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        status_layout = QVBoxLayout()
        self.posture_status = QLabel("Posture Status: Not monitoring"); self.posture_status.setAlignment(Qt.AlignCenter)
        self.posture_status.setStyleSheet(posture_style("stopped")); self.posture_status.setFont(QFont(self.app_font.family(), 12))
        status_layout.addWidget(self.posture_status)

        self.stats_display = QLabel("Detailed Status: Click 'Start Camera' to begin monitoring"); self.stats_display.setAlignment(Qt.AlignCenter); self.stats_display.setWordWrap(True)
        self.stats_display.setStyleSheet(STATUS_PANEL_QSS); self.stats_display.setFont(QFont(self.app_font.family(), 10))
        status_layout.addWidget(self.stats_display)
        layout.addLayout(status_layout)

        btn_layout = QHBoxLayout(); btn_layout.setSpacing(10); icon_size = QSize(18, 18)
        self.start_button = QPushButton("Start Camera"); self.start_button.setIcon(QIcon("assets/start_icon.png")); self.start_button.setIconSize(icon_size); self.start_button.clicked.connect(self.start_video); btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Camera"); self.stop_button.setIcon(QIcon("assets/stop_icon.png")); self.stop_button.setIconSize(icon_size); self.stop_button.clicked.connect(self.stop_video); self.stop_button.setEnabled(False); btn_layout.addWidget(self.stop_button)
        self.calibrate_button = QPushButton("Calibrate"); self.calibrate_button.setIcon(QIcon("assets/calibrate_icon.png")); self.calibrate_button.setIconSize(icon_size); self.calibrate_button.setStyleSheet(BTN_SUCCESS_QSS); self.calibrate_button.clicked.connect(self.start_calibration); btn_layout.addWidget(self.calibrate_button)
        btn_layout.addStretch(1); layout.addLayout(btn_layout)

        self.voice_status = QLabel("🎤 Voice Status: Initializing..."); self.voice_status.setAlignment(Qt.AlignCenter); self.voice_status.setStyleSheet(voice_status_style("neutral"))
        layout.addWidget(self.voice_status)

        self.live_tab.setLayout(QVBoxLayout()); self.live_tab.layout().addWidget(live_wrapper)

    # log tab
    def init_analytics_tab(self):
        layout = QVBoxLayout()
        title = QLabel("Posture Analytics")
        title.setFont(QFont("Press Start 2P", 14))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.graph_thread = GraphThread()
        layout.addWidget(self.graph_thread.canvas)

        self.graph_thread.progress.connect(
            lambda: self.graph_thread.set_tab(self.tab_widget.currentIndex())
        )
        self.graph_thread.update_plot.connect(self.graph_thread.plot_on_main_thread)

        self.graph_thread.start()

        self.tab_widget.currentChanged.connect(self.graph_thread.set_tab)

        desc = QLabel("This analytics panel will show posture metrics over time.")
        desc.setFont(QFont("Press Start 2P", 9))
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.analytics_tab.setLayout(layout)

    # settings tab
    def init_settings_tab(self):
        layout = QVBoxLayout(); layout.setSpacing(12)

        visual_group = QGroupBox("Visual Settings"); v_layout = QFormLayout()
        self.landmark_checkbox = QCheckBox("Show pose landmarks on camera feed"); self.landmark_checkbox.setChecked(self.show_landmarks); self.landmark_checkbox.stateChanged.connect(self.toggle_landmark_visibility)
        v_layout.addRow(QLabel("Landmarks:"), self.landmark_checkbox)
        landmark_info = QLabel("When enabled, shows pose detection points and connections on the video feed"); landmark_info.setStyleSheet("color: #5A6B84;")
        v_layout.addRow("", landmark_info)
        visual_group.setLayout(v_layout); layout.addWidget(visual_group)

        #detection engine group
        engine_group = QGroupBox("Detection Engine")
        e_layout = QFormLayout()

        from PyQt5.QtWidgets import QComboBox
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Rule-based (default)", "rules")
        self.engine_combo.addItem("Machine Learning (beta)", "ml")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        e_layout.addRow(QLabel("Engine:"), self.engine_combo)

        engine_help = QLabel("Switch between the handcrafted logic and the trained ML model.")
        engine_help.setStyleSheet("color: #5A6B84;")
        e_layout.addRow("", engine_help)

        engine_group.setLayout(e_layout)
        layout.addWidget(engine_group)

        notif_group = QGroupBox("Notification Settings"); n_layout = QFormLayout()
        vol_row = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal); self.volume_slider.setRange(0,100); self.volume_slider.setValue(self.notification_volume)
        self.volume_slider.setTickPosition(QSlider.TicksBelow); self.volume_slider.setTickInterval(10)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_label = QLabel(f"{self.notification_volume}%"); self.volume_label.setFixedWidth(60)
        vol_row.addWidget(self.volume_slider,1); vol_row.addWidget(self.volume_label,0,Qt.AlignRight)
        n_layout.addRow(QLabel("Notification Volume:"), vol_row)

        self.beep_interval_spinbox = QDoubleSpinBox(); self.beep_interval_spinbox.setRange(0.5,10.0); self.beep_interval_spinbox.setSingleStep(0.5)
        self.beep_interval_spinbox.setValue(self.beep_interval); self.beep_interval_spinbox.setSuffix(" seconds"); self.beep_interval_spinbox.valueChanged.connect(self._on_beep_interval_changed)
        n_layout.addRow(QLabel("Beep Interval:"), self.beep_interval_spinbox)

        self.alert_duration_spinbox = QSpinBox(); self.alert_duration_spinbox.setRange(1,60); self.alert_duration_spinbox.setValue(int(self.alert_duration))
        self.alert_duration_spinbox.setSuffix(" seconds"); self.alert_duration_spinbox.valueChanged.connect(self._on_alert_duration_changed)
        n_layout.addRow(QLabel("Alert Duration:"), self.alert_duration_spinbox)

        notif_group.setLayout(n_layout); layout.addWidget(notif_group)

        voice_group = QGroupBox("Voice Control Settings")
        voice_layout = QFormLayout()

        # Voice enable/disable
        self.voice_checkbox = QCheckBox("Enable Voice Commands")
        self.voice_checkbox.setChecked(False)
        self.voice_checkbox.stateChanged.connect(self.toggle_voice_recognition)
        voice_layout.addRow("Voice Commands:", self.voice_checkbox)

        # Language Selection
        from PyQt5.QtWidgets import QComboBox
        self.language_combo = QComboBox()
        available_languages = voice_config.get_available_languages()
        for code, name in available_languages.items():
            self.language_combo.addItem(name, code)

        # Set current language
        current_lang = voice_config.get_language()
        index = self.language_combo.findData(current_lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        voice_layout.addRow("Recognition Language:", self.language_combo)

        # Language info
        lang_info = QLabel("Speech recognition language. English fallback is automatic for other languages.")
        lang_info.setWordWrap(True)
        lang_info.setStyleSheet("color: #5A6B84; font-size: 10px;")
        voice_layout.addRow("", lang_info)

        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)

        commands_group = QGroupBox("Voice Command Bindings")
        commands_layout = QFormLayout()

        cmd_info = QLabel("Customize trigger words for each command (comma-separated)")
        cmd_info.setWordWrap(True)
        cmd_info.setStyleSheet("color: #5A6B84; font-size: 10px; margin-bottom: 10px;")
        commands_layout.addRow(cmd_info)

        self.command_inputs = {}
        command_types = [
            ("calibrate", "Calibration"),
            ("start", "Start Camera"),
            ("stop", "Stop Camera"),
            ("exit", "Exit Application"),
            ("good_posture", "Good Posture Label"),
            ("bad_posture", "Bad Posture Label"),
            ("moderate_posture", "Moderate Posture Label")
        ]

        for cmd_type, display_name in command_types:
            input_field = QLineEdit()
            current_triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(current_triggers))
            input_field.setPlaceholderText(f"Enter trigger words for {display_name}")

            input_field.editingFinished.connect(
                lambda t=cmd_type, f=input_field: self._on_command_triggers_changed(t, f.text())
            )

            self.command_inputs[cmd_type] = input_field
            commands_layout.addRow(f"{display_name}:", input_field)

        reset_commands_btn = QPushButton("Reset Commands to Defaults")
        reset_commands_btn.clicked.connect(self._reset_voice_commands)
        reset_commands_btn.setStyleSheet("padding: 6px 12px; margin-top: 10px;")
        commands_layout.addRow("", reset_commands_btn)

        commands_group.setLayout(commands_layout)
        layout.addWidget(commands_group)

        voice_help_group = QGroupBox("Voice Command Help")
        help_layout = QVBoxLayout()

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(150)
        help_content = self._generate_voice_help_text()
        help_text.setPlainText(help_content)
        help_text.setStyleSheet(
            "font-size: 10px; color: #666; padding: 8px; background-color: #f8f9fa; border-radius: 4px;"
        )
        help_layout.addWidget(help_text)

        voice_help_group.setLayout(help_layout)
        layout.addWidget(voice_help_group)
        data_group = QGroupBox("Data Management")
        data_layout = QFormLayout()

        data_btn_layout = QHBoxLayout()

        export_button = QPushButton("Export Log as CSV")
        export_button.clicked.connect(self.export_log)
        export_button.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(export_button)

        clear_log_button = QPushButton("Clear Log Data")
        clear_log_button.clicked.connect(self.clear_log)
        clear_log_button.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(clear_log_button)

        # I commented these out a while back and I have no clue if there's
        # Any reason to bring them back.
        #data_layout.addRow("Actions:", data_btn_layout)

        #data_info = QLabel("Export your posture data to CSV format or clear all logged data")
        #data_info.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        #data_layout.addRow("", data_info)

        export_voice_btn = QPushButton("📤 Export Voice Settings")
        export_voice_btn.clicked.connect(self._export_voice_settings)
        export_voice_btn.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(export_voice_btn)

        import_voice_btn = QPushButton("📥 Import Voice Settings")
        import_voice_btn.clicked.connect(self._import_voice_settings)
        import_voice_btn.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(import_voice_btn)

        data_layout.addRow("Actions:", data_btn_layout)

        data_info = QLabel("Export/import your data and voice configuration settings")
        data_info.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        data_layout.addRow("", data_info)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        settings_inner = QWidget(); settings_inner.setLayout(layout)
        settings_scroll = QScrollArea(); settings_scroll.setWidgetResizable(True); settings_scroll.setWidget(settings_inner)
        outer = QVBoxLayout(); outer.addWidget(settings_scroll); self.settings_tab.setLayout(outer)

    def _on_generate_recommendations(self):
        try:
            posture_history = getattr(backend, "get_posture_history", lambda: [])()
            references = getattr(backend, "get_recommendation_references", lambda: {})()
            focus_text = (self.issue_filter_input.text() or "").strip()
            extra_focus = [s.strip() for s in focus_text.split(",") if s.strip()]
            budget_raw = (self.budget_input.text() or "").strip()

            # Add notification (correct method signature)
            self.add_notification("💡 New product recommendations available!", "recommendation")
            self.show_toast_notification("New recommendations ready! Check the Recommendations tab.", "success")

            try:
                budget = float(budget_raw) if budget_raw else None
            except ValueError:
                budget = None

            weights = None
            issues = extra_focus if extra_focus else posture_history
            results = getattr(backend, "query_products_via_serpapi", lambda *args, **kwargs: [])(
                issues=issues, references=references, extra_focus=extra_focus, budget=budget, weights=weights
            )
            self._populate_recs_table(results)

            if hasattr(self, "stats_display"):
                self.stats_display.setText("Recommendations updated.")
        except Exception as e:
            if hasattr(self, "stats_display"):
                self.stats_display.setText(f"Failed to get recommendations: {e}")
            self.add_notification(f"❌ Failed to generate recommendations: {e}", "error")

    def _populate_recs_table(self, products):
        self.recs_table.setRowCount(0)
        if not products:
            self.recs_table.setRowCount(1); self.recs_table.setItem(0,0,QTableWidgetItem("No recommendations yet.")); return
        self.recs_table.setRowCount(len(products))
        for r, p in enumerate(products):
            title = p.get("title", "—"); cat = p.get("category", "—"); why = p.get("why", "—"); conf = p.get("confidence", None)
            price = p.get("price_text", "—"); url = p.get("url", "")
            self.recs_table.setItem(r,0,QTableWidgetItem(title))
            self.recs_table.setItem(r,1,QTableWidgetItem(cat))
            self.recs_table.setItem(r,2,QTableWidgetItem(why))
            conf_display = "—" if conf is None else f"{round(float(conf)*100):d}%"
            self.recs_table.setItem(r,3,QTableWidgetItem(conf_display))
            self.recs_table.setItem(r,4,QTableWidgetItem(price))
            self.recs_table.setItem(r,5,QTableWidgetItem(url if url else "—"))

    def _on_save_recommendations_csv(self):
        try:
            dest, _ = QFileDialog.getSaveFileName(self, "Save Recommendations", "recommendations.csv", "CSV Files (*.csv)")
            if not dest: return
            import csv
            cols = [self.recs_table.horizontalHeaderItem(i).text() for i in range(self.recs_table.columnCount())]
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(cols)
                for r in range(self.recs_table.rowCount()):
                    row = []
                    for c in range(self.recs_table.columnCount()):
                        item = self.recs_table.item(r, c); row.append(item.text() if item else "")
                    writer.writerow(row)
            if hasattr(self, "stats_display"): self.stats_display.setText("Saved recommendations to CSV.")
        except Exception as e:
            if hasattr(self, "stats_display"): self.stats_display.setText(f"Save failed: {e}")

    def toggle_landmark_visibility(self, state):
        self.show_landmarks = (state == Qt.Checked)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_landmark_visibility(self.show_landmarks)
        if hasattr(self, "stats_display"):
            self.stats_display.setText("Landmarks enabled" if self.show_landmarks else "Landmarks disabled")

    def _on_volume_changed(self, value):
        self.notification_volume = value; self.volume_label.setText(f"{value}%"); backend.update_notification_volume(value)

    def _on_beep_interval_changed(self, value):
        self.beep_interval = value; backend.update_beep_interval(value)

    def _on_alert_duration_changed(self, value):
        self.alert_duration = value; backend.update_alert_duration(value)

    def toggle_voice_recognition(self, state):
        if state == Qt.Checked:
            self.speech_thread.enable_listening()
            self.voice_status.setText("🎤 Voice Status: Listening...")
            self.voice_status.setStyleSheet(voice_status_style("on"))
        else:
            self.speech_thread.disable_listening()
            self.voice_status.setText("🎤 Voice Status: Disabled")
            self.voice_status.setStyleSheet(voice_status_style("off"))

    def handle_voice_command(self, command):
        words = command.lower().split()
        s = command.lower()
        self.voice_status.setText(f"🎤 Heard: '{command}'")

        matched_command = voice_config.match_command(command)

        if matched_command == "calibrate":
            self.start_calibration()
            self.stats_display.setText("Voice: Starting calibration...")
        elif matched_command == "stop":
            if self.video_thread.isRunning():
                self.stop_video()
                self.stats_display.setText("Voice: Camera stopped. Say 'start' to resume.")
            else:
                self.stats_display.setText("Voice: Camera already stopped.")
        elif matched_command == "exit":
            self.stats_display.setText("Voice: Exiting...")
            QTimer.singleShot(1000, self.close)
        elif matched_command == "start":
            if not self.video_thread.isRunning():
                self.start_video()
                self.stats_display.setText("Voice: Camera started.")
            else:
                self.stats_display.setText("Voice: Camera already running.")
        elif any(word in ["help", "commands", "what", "options"] for word in words):
            self.stats_display.setText("Voice: Commands: start, stop, cal, exit")
        else:
            self.stats_display.setText(f"Voice: Unknown '{command}'")

        QTimer.singleShot(4000, self.reset_voice_status)

    def update_speech_status(self, status):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText(f"Microphone status: {status}")

    def reset_voice_status(self):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText("Microphone activated: Listening...")
            self.voice_status.setStyleSheet(voice_status_style("on"))

    def toggle_notification_dropdown(self):
        if self.notification_dropdown.isVisible():
            self.animate_dropdown_close()
            return

        self.notification_dropdown.adjustSize()  # let layout compute size
        self.notification_dropdown.setMaximumHeight(0)  # for your open animation start

        btn_bottom_left_global = self.notification_btn.mapToGlobal(self.notification_btn.rect().bottomLeft())

        dropdown_w = self.notification_dropdown.width()
        dropdown_x = btn_bottom_left_global.x() + self.notification_btn.width() - dropdown_w
        dropdown_y = btn_bottom_left_global.y() + 5  # small offset below button

        self.notification_dropdown.move(dropdown_x, dropdown_y)

        self.notification_dropdown.show()
        self.notification_dropdown.setFocus(Qt.PopupFocusReason)
        self.animate_dropdown_open()

    # I really botched this, but I don't think these animations work at *all*
    def animate_dropdown_open(self):

        if self.notification_dropdown.height() > 0:
            return

        self.notification_dropdown.setMaximumHeight(0)

        animation = QPropertyAnimation(self.notification_dropdown, b"maximumHeight")
        animation.setDuration(200)
        animation.setStartValue(0)
        animation.setEndValue(400)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        self._active_animations.append(animation)
        animation.finished.connect(lambda: self._cleanup_animation(animation))
        animation.start()

    def animate_dropdown_close(self):
        animation = QPropertyAnimation(self.notification_dropdown, b"maximumHeight")
        animation.setDuration(200)
        animation.setStartValue(self.notification_dropdown.height())
        animation.setEndValue(0)
        animation.setEasingCurve(QEasingCurve.InCubic)
        animation.finished.connect(self.notification_dropdown.hide)

        self._active_animations.append(animation)
        animation.finished.connect(lambda: self._cleanup_animation(animation))
        animation.start()

    def _cleanup_animation(self, animation):
        try:
            self._active_animations.remove(animation)
        except ValueError:
               pass

    def should_send_notification(self):
        current_time = datetime.now()
        if self.last_notification_time is None:
            self.last_notification_time = current_time
            return True

        time_diff = (current_time - self.last_notification_time).seconds
        if time_diff >= self.posture_notification_cooldown:
            self.last_notification_time = current_time
            return True
        return False

    def add_notification(self, text, notification_type="info"):
        self.notification_dropdown.add_notification(text, notification_type)
        self.notification_btn.set_unread_count(self.notification_dropdown.get_unread_count())

        if hasattr(self, 'notification_sound_enabled') and self.notification_sound_enabled:
            QApplication.beep()

    def show_toast_notification(self, message, toast_type="info"):
        # Not sure how I feel about the toasts.
        # It's a user-friendly way to show if you're calibrating or doing something or
        # Just general feedback stuff that isn't within our camera
        # But I'm not a fan of it. Maybe if I align it to a corner, because the current
        # centered toast is lame.
        try:
            toast = QFrame(self)
            toast.setAttribute(Qt.WA_DeleteOnClose)
            toast.setObjectName(f"toast_{id(toast)}")  # Unique identifier

            if not hasattr(self, '_active_toasts'):
                self._active_toasts = []
            self._active_toasts.append(toast)

            styles = {
                "info": "background-color: #17A2B8;",
                "success": "background-color: #28A745;",
                "warning": "background-color: #FFC107; color: #000;",
                "error": "background-color: #DC3545;"
            }

            toast.setStyleSheet(f"""
                QFrame {{
                    {styles.get(toast_type, styles["info"])}
                    border-radius: 8px;
                    padding: 12px 20px;
                }}
            """)

            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setXOffset(0)
            shadow.setYOffset(2)
            shadow.setColor(QColor(0, 0, 0, 80))
            toast.setGraphicsEffect(shadow)

            layout = QHBoxLayout(toast)
            label = QLabel(message)
            label.setStyleSheet("""
                color: white; 
                font-size: 12px; 
                font-family: 'Segoe UI', sans-serif;
            """ if toast_type != "warning" else """
                color: black; 
                font-size: 12px; 
                font-family: 'Segoe UI', sans-serif;
            """)
            layout.addWidget(label)

            # Changing this later
            toast.adjustSize()
            x = (self.width() - toast.width()) // 2
            y = 80  # Below toolbar lol
            toast.move(x, y)
            toast.show()
            toast.raise_()

            toast.setWindowOpacity(0)
            fade_in = QPropertyAnimation(toast, b"windowOpacity")
            fade_in.setDuration(200)
            fade_in.setStartValue(0)
            fade_in.setEndValue(0.95)

            if not hasattr(self, '_active_animations'):
                self._active_animations = []
            self._active_animations.append(fade_in)

            fade_in.start()

            # Remove after 3 seconds. Will probably tie this to a setting in the future
            # God I love settings
            QTimer.singleShot(3000, lambda: self._safe_remove_toast(toast))

        except Exception as e:
            print(f"[ERROR] show_toast_notification failed: {e}")

    def _safe_remove_toast(self, toast):
        try:
            if toast in self._active_toasts:
                self._active_toasts.remove(toast)

            if toast and not toast.isHidden():
                fade_out = QPropertyAnimation(toast, b"windowOpacity")
                fade_out.setDuration(200)
                fade_out.setStartValue(toast.windowOpacity())
                fade_out.setEndValue(0)

                fade_out.finished.connect(lambda t=toast: self._delete_toast(t))

                if not hasattr(self, '_active_animations'):
                    self._active_animations = []
                self._active_animations.append(fade_out)

                fade_out.start()
        except RuntimeError:
            pass

    def _delete_toast(self, toast):
        try:
            if toast in self._active_toasts:
                self._active_toasts.remove(toast)
            toast.deleteLater()
        except:
            pass

    def fade_out_toast(self, toast):
        if not toast or toast.isHidden():
            return

        try:
            animation = QPropertyAnimation(toast, b"windowOpacity")
            animation.setDuration(200)
            animation.setStartValue(toast.windowOpacity())
            animation.setEndValue(0)
            animation.finished.connect(toast.deleteLater)

            self._active_animations.append(animation)
            animation.finished.connect(lambda: self._cleanup_animation(animation))
            animation.start()
        except RuntimeError:
            pass

    def _on_language_changed(self, index):
        language_code = self.language_combo.itemData(index)
        if voice_config.set_language(language_code):
            language_name = self.language_combo.itemText(index)
            self.stats_display.setText(f"Language changed to {language_name}")
            if self.speech_thread.isRunning() and self.voice_checkbox.isChecked():
                self.speech_thread.disable_listening()
                QTimer.singleShot(500, lambda: self.speech_thread.enable_listening())
        else:
            self.stats_display.setText("Failed to change language")

    def _on_command_triggers_changed(self, command_type, text):
        triggers = [t.strip() for t in text.split(',') if t.strip()]
        if triggers:
            voice_config.set_command_triggers(command_type, triggers)
            self.stats_display.setText(f"Updated {command_type} triggers")

    def _reset_voice_commands(self):
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            'Restore Default Voice Commands [ENG]',
            'Restore to default commands? This cannot be undone!',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:

            voice_config.config["commands"] = voice_config.DEFAULT_CONFIG["commands"].copy()
            voice_config.save_config()

        for cmd_type, input_field in self.command_inputs.items():
            triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(triggers))

        self.stats_display.setText("Voice commands reset to defaults")

    # This is sincerely outdated and may not even need to exist anymore
    # Debating if it should stay and be dynamically updated
    # To hold currently created voice binds
    # But... user can just look at their binds to know what they can say
    def _generate_voice_help_text(self):
        help_lines = ["Current Voice Commands:\n"]

        command_names = {
            "calibrate": "Calibration",
            "start": "Start Camera",
            "stop": "Stop Camera",
            "exit": "Exit Application",
            "good_posture": "Good Posture",
            "bad_posture": "Bad Posture",
            "moderate_posture": "Moderate Posture"
        }

        for cmd_type, display_name in command_names.items():
            triggers = voice_config.get_command_triggers(cmd_type)
            if triggers:
                examples = triggers[:3]
                examples_text = ', '.join(f'"{t}"' for t in examples)
                if len(triggers) > 3:
                    examples_text += f" (and {len(triggers) - 3} more)"
                help_lines.append(f"• {display_name}: {examples_text}")

        help_lines.append("\nTips:")
        help_lines.append("• Speak clearly at normal volume")
        help_lines.append("• Wait for 'Listening...' before speaking")
        help_lines.append(
            f"• Current language: {voice_config.config['language_options'].get(voice_config.get_language(), 'Unknown')}")

        return "\n".join(help_lines)

    def _export_voice_settings(self):
        from PyQt5.QtWidgets import QFileDialog
        import json

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Voice Settings",
            "voice_settings_backup.json",
            "JSON Files (*.json)"
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(voice_config.config, f, indent=2, ensure_ascii=False)
                self.stats_display.setText(f"Voice settings exported successfully")
            except Exception as e:
                self.stats_display.setText(f"Export failed: {e}")

    def _import_voice_settings(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import json

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Voice Settings",
            "",
            "JSON Files (*.json)"
        )

        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    imported_config = json.load(f)

                if "commands" in imported_config and "language" in imported_config:
                    voice_config.config.update(imported_config)
                    voice_config.save_config()

                    self._refresh_voice_settings_ui()

                    self.stats_display.setText("Voice settings imported successfully")
                else:
                    QMessageBox.warning(self, "Import Error", "Invalid voice settings file")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import: {e}")

    def _refresh_voice_settings_ui(self):
        current_lang = voice_config.get_language()
        index = self.language_combo.findData(current_lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)


        for cmd_type, input_field in self.command_inputs.items():
            triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(triggers))

        if hasattr(self, 'voice_help_text'):
            self.voice_help_text.setPlainText(self._generate_voice_help_text())

    def start_calibration(self):
        backend.calibration_start_time = time.time()
        backend.is_calibrating = True
        backend.calibration_data = {k: [] for k in backend.calibration_data}
        self.stats_display.setText("Calibration started. Hold posture 8s...")

        self.add_notification("Calibration started - hold good posture for 8 seconds", "calibration")
        self.show_toast_notification("Calibration started! Hold your best posture.", "info")

    # logs
    def load_log(self):
        try:
            log_path = os.path.join(os.path.dirname(__file__), "posture_trend_log.csv")
            if os.path.exists(log_path):
                expected = ["Timestamp","Mode","Facing","Posture Status","Head Tilt","Confidence Score"]
                df = pd.read_csv(log_path, header=None, names=expected)
                display_df = df[expected].tail(50).reset_index(drop=True)
                self.log_table.setColumnCount(len(expected))
                self.log_table.setHorizontalHeaderLabels(expected)
                self.log_table.setRowCount(len(display_df))
                for r, row in display_df.iterrows():
                    for c, name in enumerate(expected):
                        v = row[name]; v = "—" if pd.isna(v) else v
                        item = QTableWidgetItem(str(v))
                        if name == "Posture Status":
                            label = str(v).lower()
                            if "bad" in label: item.setForeground(QColor("#C62828"))
                            elif "good" in label: item.setForeground(QColor("#2E7D32"))
                            elif "moderate" in label: item.setForeground(QColor("#B26A00"))
                        self.log_table.setItem(r, c, item)
            else:
                self.log_table.setRowCount(1); self.log_table.setColumnCount(1); self.log_table.setItem(0,0,QTableWidgetItem("📂 No log file found."))
        except Exception as e:
            self.log_table.setRowCount(1); self.log_table.setColumnCount(1); self.log_table.setItem(0,0,QTableWidgetItem(f"❌ Error loading log: {e}"))

    # devs popup
    def expand_folder_popup(self):
        if hasattr(self, 'folder_popup') and self.folder_popup.isVisible(): return
        self.folder_popup = QFrame(self); self.folder_popup.setStyleSheet(POPUP_FRAME_QSS); self.folder_popup.setVisible(True); self.folder_popup.raise_()
        popup_layout = QVBoxLayout(self.folder_popup); popup_layout.setContentsMargins(20,20,20,20); popup_layout.setSpacing(10)
        header_layout = QGridLayout()
        self.close_button = QPushButton("✖"); self.close_button.setFixedSize(30,30)
        self.close_button.setStyleSheet("QPushButton { background-color: #E55353; color: white; border: none; border-radius: 6px; } QPushButton:hover { background-color: #C94444; }")
        self.close_button.clicked.connect(self.folder_popup.close)
        self.devs_title = QLabel("Meet the Devs of Spinewise"); self.devs_title.setFont(QFont(self.app_font.family(), 14, QFont.DemiBold)); self.devs_title.setAlignment(Qt.AlignCenter); self.devs_title.setStyleSheet("color: #0B5CAD;")
        header_layout.addWidget(self.devs_title, 0, 1, alignment=Qt.AlignHCenter)
        header_layout.addWidget(self.close_button, 0, 2, alignment=Qt.AlignRight)
        header_layout.setColumnStretch(0,1); header_layout.setColumnStretch(1,2); header_layout.setColumnStretch(2,1)
        popup_layout.addLayout(header_layout)

        self.carousel_widget = QStackedWidget(); self.carousel_widget.setFixedHeight(int(self.height()*0.6))

        devs_info = [
            ("Emdya Permuy-Llovio ", "Product Manager", "assets/dev1.png", "https://www.linkedin.com/in/emdyapermuy/", "https://github.com/Emdya"),
            ("Juan Mieses", "Fullstack Development ", "assets/dev2.png", "https://www.linkedin.com/in/juanmieses003/", "https://github.com/Jmies-27"),
            ("Javier Brasil", "Fullstack Development", "assets/dev3.png", "https://www.linkedin.com/in/javier-a-brasil/", "https://github.com/SlyyJavii"),
            ("John Pena ", "Backend and Machine Learning Development", "assets/dev4.png", "https://www.linkedin.com/in/johnpenacs/", "https://github.com/jpena173"),
            ("Jake Rodriguez", "Visual and Audio Alert System", "assets/dev5.png", "https://www.linkedin.com/in/jake-rodriguez-917a24142/","https://github.com/jrodr995"),
        ]
        captions = [
            "Emdya Permuy-Llovio is an Undergraduate BS in Computer Science student at Florida International University... ",
            "Juan A. Mieses is a Florida International University Undergraduate student pursuing a Bachelor's Degree... ",
            "Javier builds solid infrastructure and efficient code.",
            "John Pena is an aspiring undergraduate studying Computer Science, preferring cybersecurity tasks and backend development...",
            "Jake crafts beautiful alert systems and UI animations."
        ]

        for idx, (name, role, img_path, linkedin, github) in enumerate(devs_info):
            card = QWidget(); card_layout = QVBoxLayout(card); card_layout.setContentsMargins(30,10,30,10); card_layout.setSpacing(12); card_layout.setAlignment(Qt.AlignCenter)
            card.setStyleSheet("background-color: #FFFFFF; border: 1px solid #D5E3F4; border-radius: 12px;")
            image = QLabel(); image.setPixmap(QPixmap(img_path).scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)); image.setAlignment(Qt.AlignCenter); card_layout.addWidget(image)
            name_label = QLabel(name); name_label.setFont(QFont(self.app_font.family(), 12, QFont.Medium)); name_label.setAlignment(Qt.AlignCenter); name_label.setStyleSheet("color: #0F2238; padding: 8px"); card_layout.addWidget(name_label)

            role_row = QHBoxLayout(); role_row.setAlignment(Qt.AlignCenter)
            github_button = QPushButton(); github_button.setCursor(Qt.PointingHandCursor); github_button.setIcon(QIcon("assets/icons/GitHub_Invertocat_Dark.png"))
            github_button.setIconSize(QSize(20,20)); github_button.setFixedSize(28,28); github_button.setStyleSheet("QPushButton { background: transparent; border: none; } QPushButton:hover { background: #F0F4FF; border-radius: 6px; }")
            github_button.clicked.connect(lambda _, url=github: QDesktopServices.openUrl(QUrl(url))); role_row.addWidget(github_button)

            role_label = QLabel(role); role_label.setStyleSheet("color: #5A6B84;"); role_row.addWidget(role_label)

            linkedin_button = QPushButton(); linkedin_button.setCursor(Qt.PointingHandCursor); linkedin_button.setIcon(QIcon("assets/icons/LinkedIn_logo_initials.png"))
            linkedin_button.setIconSize(QSize(20,20)); linkedin_button.setFixedSize(28,28); linkedin_button.setStyleSheet("QPushButton { background: transparent; border: none; } QPushButton:hover { background: #F0F4FF; border-radius: 6px; }")
            linkedin_button.clicked.connect(lambda _, url=linkedin: QDesktopServices.openUrl(QUrl(url))); role_row.addWidget(linkedin_button)

            card_layout.addLayout(role_row)

            caption_label = QLabel(captions[idx]); caption_label.setWordWrap(True); caption_label.setAlignment(Qt.AlignCenter); caption_label.setStyleSheet("color: #2E3C51;"); card_layout.addWidget(caption_label)
            self.carousel_widget.addWidget(card)

        popup_layout.addWidget(self.carousel_widget)

        nav_layout = QHBoxLayout(); nav_layout.setAlignment(Qt.AlignCenter)
        left_btn = QPushButton("◀"); left_btn.setFixedSize(30,30); left_btn.setStyleSheet("QPushButton { background: #EAF2FF; color: #0F2238; border-radius: 15px; } QPushButton:hover { background: #DDEBFF; }")
        left_btn.clicked.connect(lambda: self.carousel_widget.setCurrentIndex((self.carousel_widget.currentIndex() - 1) % self.carousel_widget.count()))
        nav_layout.addWidget(left_btn)

        dot_group = QButtonGroup(); self.pagination_dots = []
        for i in range(self.carousel_widget.count()):
            dot = QRadioButton(); dot.setStyleSheet(DOT_QSS)
            dot.toggled.connect(lambda checked, idx=i: self.carousel_widget.setCurrentIndex(idx) if checked else None)
            dot_group.addButton(dot); self.pagination_dots.append(dot); nav_layout.addWidget(dot)

        right_btn = QPushButton("▶"); right_btn.setFixedSize(30,30); right_btn.setStyleSheet("QPushButton { background: #EAF2FF; color: #0F2238; border-radius: 15px; } QPushButton:hover { background: #DDEBFF; }")
        right_btn.clicked.connect(lambda: self.carousel_widget.setCurrentIndex((self.carousel_widget.currentIndex() + 1) % self.carousel_widget.count()))
        nav_layout.addWidget(right_btn)
        popup_layout.addLayout(nav_layout)

        if self.pagination_dots: self.pagination_dots[0].setChecked(True)
        def sync_dots(index):
            if 0 <= index < len(self.pagination_dots): self.pagination_dots[index].setChecked(True)
        self.carousel_widget.currentChanged.connect(sync_dots)

        start_pos = self.folder_icon.mapToGlobal(self.folder_icon.rect().center()); start_pos = self.mapFromGlobal(start_pos)
        self.folder_popup.setGeometry(QRect(start_pos.x(), start_pos.y(), 10, 10))
        end_w = int(self.width() * 0.8); end_h = int(self.height() * 0.7); end_x = int((self.width() - end_w) / 2); end_y = int((self.height() - end_h) / 2)
        self.popup_anim = QPropertyAnimation(self.folder_popup, b"geometry"); self.popup_anim.setDuration(400)
        self.popup_anim.setStartValue(self.folder_popup.geometry()); self.popup_anim.setEndValue(QRect(end_x, end_y, end_w, end_h)); self.popup_anim.setEasingCurve(QEasingCurve.OutCubic); self.popup_anim.start()

    # data mgmt
    def clear_log(self):
        try:
            if os.path.exists("posture_trend_log.csv"):
                os.remove("posture_trend_log.csv"); self.log_table.setRowCount(1); self.log_table.setItem(0,0,QTableWidgetItem("🗑️ Log data cleared."))
            else:
                self.log_table.setRowCount(1); self.log_table.setItem(0,0,QTableWidgetItem("📂 No log file to clear."))
        except Exception as e:
            print("[ERROR] clear_log:", e)

    def export_log(self):
        if os.path.exists("posture_trend_log.csv"):
            dest, _ = QFileDialog.getSaveFileName(self, "Save Log", "posture_trend_log.csv", "CSV Files (*.csv)")
            if dest:
                try:
                    with open("posture_trend_log.csv", "r") as src, open(dest, "w") as dst: dst.write(src.read())
                    self.stats_display.setText("Log exported.")
                except Exception as e:
                    self.stats_display.setText(f"Export failed: {e}")
        else:
            self.stats_display.setText("No log file to export")

    def start_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread._run_flag = False
            self.video_thread.wait(1000)

        self.video_thread = VideoThread(show_landmarks=self.show_landmarks)

        self.video_thread.change_pixmap_signal.connect(
            self.update_image, Qt.QueuedConnection
        )
        self.video_thread.update_stats_signal.connect(
            self.update_stats, Qt.QueuedConnection
        )

        self.video_thread.start()

        self.add_notification(f" Camera turned on at {datetime.now().strftime('%I:%M %p')}!", "info")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stats_display.setText("Camera starting...")
        self.image_label.setMinimumSize(800, 480)
        self.image_label.setText("")
        self.posture_status.setText("Monitoring Posture...")
        self.posture_status.setStyleSheet(posture_style("monitor"))

    def stop_video(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            if not self.video_thread.wait(2000):  # Wait 2 seconds
                print("[WARNING] Video thread went oopsie! Go tell Juan.")

        self.update_ui_after_stop()

    def finish_video_stop(self):
        if self.video_thread and self.video_thread.isRunning():
            QTimer.singleShot(500, self.force_video_stop)
        else:
            self.update_ui_after_stop()

    def force_video_stop(self):
        if self.video_thread and self.video_thread.isRunning():
            try: self.video_thread.terminate()
            except Exception: pass
        self.update_ui_after_stop()

    def update_ui_after_stop(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stats_display.setText("Camera stopped")
        self.posture_status.setText("Camera Stopped")
        self.posture_status.setStyleSheet(posture_style("stopped"))
        self.image_label.setText("Click 'Start Camera' to begin webcam feed")
        self.image_label.clear()

    def check_app_status(self):
        self.stats_display.setText("⏹️ Camera stopped - App is running")

    def eventFilter(self, source, event):
        if source == self.folder_icon:
            if event.type() == QEvent.Enter:
                self.folder_icon.setPixmap(QPixmap("assets/icons/folder_open.png"))
            elif event.type() == QEvent.Leave:
                self.folder_icon.setPixmap(QPixmap("assets/icons/folder_closed.png"))
            elif event.type() == QEvent.MouseButtonPress:
                self.expand_folder_popup()
        return super().eventFilter(source, event)

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def update_stats(self, text):
        # Currently the most evil part of the code
        # Changing this in accordance with the new notification system has been
        # A nightmare of errors and crashes. Honestly, it's the primary reason
        # this experimental branch exits.
        try:
            if threading.current_thread() != threading.main_thread():
                QTimer.singleShot(0, lambda: self.update_stats(text))
                return

            current_text = self.stats_display.text()
            if "Voice" not in current_text:
                self.stats_display.setText(f"Analysis: {text}")

            if text and not any(
                    w in text.lower() for w in
                    ["detecting", "stabilizing", "transitioning", "confirming", "analyzing"]):
                tl = text.lower()

                current_time = datetime.now()

                if "good posture" in tl:
                    self.posture_status.setText("Good Posture")
                    self.posture_status.setStyleSheet(posture_style("good"))

                    if hasattr(self, 'last_posture_state') and self.last_posture_state in ["bad", "moderate"]:
                        if hasattr(self, 'bad_posture_start_time') and self.bad_posture_start_time:
                            duration = (current_time - self.bad_posture_start_time).seconds
                            if duration > 30:  # Only notify if bad posture lasted > 30 seconds
                                minutes = duration // 60
                                seconds = duration % 60
                                if minutes > 0:
                                    msg = f"✅ Posture corrected! Was bad for {minutes}m {seconds}s"
                                else:
                                    msg = f"✅ Posture corrected! Was bad for {seconds} seconds"

                                QTimer.singleShot(0, lambda: self.add_notification(msg, "good_posture"))
                                QTimer.singleShot(0, lambda: self.show_toast_notification(msg, "success"))

                            self.bad_posture_start_time = None

                    self.last_posture_state = "good"

                elif "moderately bad posture" in tl or "moderate" in tl:
                    self.posture_status.setText("Moderate Posture Issues")
                    self.posture_status.setStyleSheet(posture_style("moderate"))

                    if not hasattr(self, 'last_posture_state'):
                        self.last_posture_state = "good"

                    if self.last_posture_state == "good":
                        self.bad_posture_start_time = current_time
                        if self.should_send_notification():
                            msg = f"⚠️ Moderate posture issues detected at {current_time.strftime('%I:%M %p')}"
                            QTimer.singleShot(0, lambda: self.add_notification(msg, "bad_posture"))

                    self.last_posture_state = "moderate"

                elif "bad posture" in tl:
                    self.posture_status.setText("Poor Posture Detected")
                    self.posture_status.setStyleSheet(posture_style("bad"))

                    if not hasattr(self, 'last_posture_state'):
                        self.last_posture_state = "good"

                    if self.last_posture_state in ["good", "moderate"]:
                        if self.last_posture_state == "good":
                            self.bad_posture_start_time = current_time

                        if self.should_send_notification():
                            msg = f"🚨 Bad posture detected at {current_time.strftime('%I:%M %p')}"
                            QTimer.singleShot(0, lambda: self.add_notification(msg, "bad_posture"))
                            QTimer.singleShot(0, lambda: self.show_toast_notification(
                                "Bad posture detected! Please adjust your position.", "warning"
                            ))

                    self.last_posture_state = "bad"

                elif "no pose" in tl:
                    self.posture_status.setText("No Person Detected")
                    self.posture_status.setStyleSheet(posture_style("stopped"))

                    if hasattr(self, 'last_posture_state') and self.last_posture_state in ["bad", "moderate"]:
                        if hasattr(self, 'bad_posture_start_time') and self.bad_posture_start_time:
                            duration = (current_time - self.bad_posture_start_time).seconds
                            if duration > 60:
                                msg = f"ℹ️ Person left view during posture monitoring"
                                QTimer.singleShot(0, lambda: self.add_notification(msg, "info"))

                    self.last_posture_state = "no_pose"
                    self.bad_posture_start_time = None

        except Exception as e:
            print(f"[ERROR] update_stats failed: {e}")

    def closeEvent(self, event):
        print("[APP] Shutting down...")

        # Kill every single thread. This is *not* a dummy fix.
        # Every single one of these need to be violently interrupted and cleared before
        # The program stops. Why? I'm sick and tired of getting that
        # -1073740791 (0xC0000409) Stack buffer overrun error.
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(2000)

        if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
            self.speech_thread.stop()
            self.speech_thread.wait(2000)

        if hasattr(self, 'graph_thread') and self.graph_thread.isRunning():
            self.graph_thread.stop()
            self.graph_thread.wait(2000)

        # Also killing all of our animations related to notifications.
        if hasattr(self, '_active_animations'):
            for animation in self._active_animations[:]:
                try:
                    animation.stop()
                except:
                    pass
            self._active_animations.clear()

        # Throwing this in terminal just to doublecheck
        # Usually it won't throw when -1073740791 (0xC0000409) happens
        event.accept()
        print("[APP] Shutdown complete")


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())