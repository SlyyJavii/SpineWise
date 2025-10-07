import os, queue, cv2, time, numpy as np, pandas as pd, mediapipe as mp, threading, backend, speech_recognition as sr
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QStackedWidget, QButtonGroup, QRadioButton, QSizePolicy, QFrame,
    QVBoxLayout, QWidget, QTabWidget, QMainWindow, QFileDialog, QTextEdit, QDoubleSpinBox,
    QScrollArea, QSpinBox, QHBoxLayout, QCheckBox, QFormLayout, QSlider, QGroupBox, QProgressBar,
    QTableWidgetItem, QTableWidget, QGridLayout, QHeaderView, QAction, QLineEdit
)
from PyQt5.QtGui import QImage, QDesktopServices, QPixmap, QFont, QIcon, QFontDatabase, QPalette, QBrush, QPainter, QColor
from PyQt5.QtCore import Qt, QUrl, QSize, QPropertyAnimation, QRect, QEasingCurve, QThread, pyqtSignal, QEvent, QTimer

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
                    command = None
                    try:
                        command = self.recognizer.recognize_google(audio, language='en-US').lower().strip()
                    except Exception:
                        try:
                            command = self.recognizer.recognize_google(audio, language='en', show_all=False).lower().strip()
                        except Exception:
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
        self.wait()

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
        if not cap.isOpened(): return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        try:
            while self._run_flag:
                ret, frame = cap.read()
                if not ret: continue
                self.raw_queue.put(frame)
                processed_frame = self.processed_queue.get()
                self.change_pixmap_signal.emit(processed_frame)
        except Exception:
            pass
        finally:
            cap.release()

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

        self.tab_widget = QTabWidget(); self.tab_widget.setElideMode(Qt.ElideRight)
        main_layout = QVBoxLayout(); main_layout.setAlignment(Qt.AlignTop); main_layout.setContentsMargins(16,16,16,16); main_layout.addWidget(self.tab_widget)
        central = QWidget(); central.setLayout(main_layout); self.setCentralWidget(central)

        menu = self.menuBar(); view_menu = menu.addMenu("View")
        rec_action = QAction("Recommendations", self); rec_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.recommendations_tab)); view_menu.addAction(rec_action)

        self.live_tab = QWidget(); self.log_tab = QWidget(); self.settings_tab = QWidget(); self.recommendations_tab = QWidget()
        self.tab_widget.addTab(self.live_tab, "Live Posture")
        self.tab_widget.addTab(self.log_tab, "Posture Log")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.recommendations_tab, "Recommendations")

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
        self.init_log_tab()
        self.init_settings_tab()
        self.init_recommendations_tab()
        self.speech_thread.start()

    # recommendations tab
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
    def init_log_tab(self):
        layout = QVBoxLayout(); layout.setSpacing(12)
        title = QLabel("Posture Data Log"); title.setFont(QFont(self.app_font.family(), 16, QFont.DemiBold)); title.setStyleSheet("color: #0B5CAD;"); layout.addWidget(title)
        self.log_table = QTableWidget(); self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels(["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.log_table)
        btn_layout = QHBoxLayout()
        load_button = QPushButton("📊 Load Posture Log"); load_button.clicked.connect(self.load_log); btn_layout.addWidget(load_button)
        refresh_button = QPushButton("🔄 Refresh"); refresh_button.clicked.connect(self.load_log); btn_layout.addWidget(refresh_button)
        btn_layout.addStretch(1); layout.addLayout(btn_layout)
        self.log_tab.setLayout(layout)

    # settings tab
    def init_settings_tab(self):
        layout = QVBoxLayout(); layout.setSpacing(12)

        visual_group = QGroupBox("Visual Settings"); v_layout = QFormLayout()
        self.landmark_checkbox = QCheckBox("Show pose landmarks on camera feed"); self.landmark_checkbox.setChecked(self.show_landmarks); self.landmark_checkbox.stateChanged.connect(self.toggle_landmark_visibility)
        v_layout.addRow(QLabel("Landmarks:"), self.landmark_checkbox)
        landmark_info = QLabel("When enabled, shows pose detection points and connections on the video feed"); landmark_info.setStyleSheet("color: #5A6B84;")
        v_layout.addRow("", landmark_info)
        visual_group.setLayout(v_layout); layout.addWidget(visual_group)

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

        voice_title = QLabel("🎤 Voice Control Settings"); voice_title.setFont(QFont(self.app_font.family(), 12, QFont.Medium)); voice_title.setStyleSheet("color: #0B5CAD;"); layout.addWidget(voice_title)
        voice_group = QGroupBox(); vg_layout = QVBoxLayout()
        self.voice_checkbox = QCheckBox("Enable Voice Commands"); self.voice_checkbox.setChecked(False); self.voice_checkbox.stateChanged.connect(self.toggle_voice_recognition); vg_layout.addWidget(self.voice_checkbox)
        voice_help_group = QGroupBox("Voice Commands Available"); vh_layout = QVBoxLayout()
        voice_help_label = QLabel(
            "Camera Control:\n"
            "• \"stop\" only turns off camera – \"start\" turns it back on\n"
            "• App keeps running when camera is stopped\n"
            "• Only \"exit\" will close the entire application\n\n"
            "Speech Tips:\n"
            "• Use \"stop\" to pause camera, \"exit\" to close app\n"
            "• \"cal\" works better than \"calibrate\"\n"
            "• Wait for \"Listening...\" before speaking\n"
            "• Speak clearly at normal volume\n\n"
            "Examples:\n"
            "• \"stop\" → Camera off\n"
            "• \"start\" → Camera on\n"
            "• \"exit\" → Close app"
        )
        voice_help_label.setWordWrap(True); vh_layout.addWidget(voice_help_label); voice_help_group.setLayout(vh_layout)
        vg_layout.addWidget(voice_help_group); voice_group.setLayout(vg_layout); layout.addWidget(voice_group)

        data_section = QLabel("📁 Data Management"); data_section.setFont(QFont(self.app_font.family(), 12, QFont.Medium))
        data_section.setStyleSheet("QLabel { color: #0F2238; background: #FFFFFF; border: 1px solid #D5E3F4; border-radius: 8px; padding: 8px; }")
        layout.addWidget(data_section)

        data_btns = QHBoxLayout()
        export_button = QPushButton("💾 Export Log as CSV"); export_button.clicked.connect(self.export_log); data_btns.addWidget(export_button)
        clear_log_button = QPushButton("🗑️ Clear Log Data"); clear_log_button.clicked.connect(self.clear_log); data_btns.addWidget(clear_log_button)
        data_btns.addStretch(1); layout.addLayout(data_btns)

        info_label = QLabel(
            "Instructions:\n"
            "1. Enable voice commands\n"
            "2. Start the camera feed\n"
            "3. Say \"calibrate\" or click to set baseline\n"
            "4. Maintain posture for 8 seconds\n"
            "5. Monitor and receive alerts\n"
            "6. View data in 'Posture Log'\n\n"
            "Tips:\n"
            "• Good lighting and quiet room\n"
            "• Mic permissions enabled\n"
            "• Speak clearly\n"
            "• Stable camera position"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #2E3C51; background: #FFFFFF; border: 1px solid #D5E3F4; border-radius: 10px; padding: 12px; }")
        info_scroll = QScrollArea(); info_scroll.setWidgetResizable(True); info_scroll.setFixedHeight(200); info_scroll.setWidget(info_label)
        layout.addWidget(info_scroll)
        layout.addStretch(1)

        settings_inner = QWidget(); settings_inner.setLayout(layout)
        settings_scroll = QScrollArea(); settings_scroll.setWidgetResizable(True); settings_scroll.setWidget(settings_inner)
        outer = QVBoxLayout(); outer.addWidget(settings_scroll); self.settings_tab.setLayout(outer)

    # recs logic
    def _on_generate_recommendations(self):
        try:
            posture_history = getattr(backend, "get_posture_history", lambda: [])()
            references = getattr(backend, "get_recommendation_references", lambda: {})()
            focus_text = (self.issue_filter_input.text() or "").strip()
            extra_focus = [s.strip() for s in focus_text.split(",") if s.strip()]
            budget_raw = (self.budget_input.text() or "").strip()
            try: budget = float(budget_raw) if budget_raw else None
            except ValueError: budget = None
            weights = None
            issues = extra_focus if extra_focus else posture_history
            results = getattr(backend, "query_products_via_serpapi", lambda *args, **kwargs: [])(
                issues=issues, references=references, extra_focus=extra_focus, budget=budget, weights=weights
            )
            self._populate_recs_table(results)
            if hasattr(self, "stats_display"): self.stats_display.setText("Recommendations updated.")
        except Exception as e:
            if hasattr(self, "stats_display"): self.stats_display.setText(f"Failed to get recommendations: {e}")

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

    # settings handlers
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

    # voice
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
        calibration_triggers = ["calibrate","calibration","collab","cal","caliber","collaborate","calib","kelly","cali","cab","start calibration"]
        if any(t in s for t in calibration_triggers):
            self.start_calibration(); self.stats_display.setText("Voice: Starting calibration...")
        elif any(p in s for p in ["stop camera","pause camera","turn off camera","camera off"]) or (len(words)==1 and words[0] in ["stop","pause","halt","off"]):
            if self.video_thread.isRunning():
                self.stop_video(); self.stats_display.setText("Voice: Camera stopped. Say 'start' to resume.")
            else:
                self.stats_display.setText("Voice: Camera already stopped.")
        elif any(p in s for p in ["exit","quit","close app","goodbye","end app","close application","shut down"]):
            self.stats_display.setText("Voice: Exiting..."); QTimer.singleShot(1000, self.close)
        elif any(word in ["start","begin","go","play","run","on"] for word in words) or any(p in s for p in ["turn on","start camera","begin camera"]):
            if not self.video_thread.isRunning():
                self.start_video(); self.stats_display.setText("Voice: Camera started.")
            else:
                self.stats_display.setText("Voice: Camera already running.")
        elif any(word in ["help","commands","what","options"] for word in words):
            self.stats_display.setText("Voice: Commands: start, stop, cal, exit")
        else:
            self.stats_display.setText(f"Voice: Unknown '{command}'")
        QTimer.singleShot(4000, self.reset_voice_status)

    def update_speech_status(self, status):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText(f"🎤 Voice Status: {status}")

    def reset_voice_status(self):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText("🎤 Voice Status: Listening...")
            self.voice_status.setStyleSheet(voice_status_style("on"))

    # calibration
    def start_calibration(self):
        backend.calibration_start_time = time.time()
        backend.is_calibrating = True
        backend.calibration_data = {k: [] for k in backend.calibration_data}
        self.stats_display.setText("Calibration started. Hold posture 8s...")

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

    # camera control
    def start_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread._run_flag = False; self.video_thread.wait(1000)
        self.video_thread = VideoThread(show_landmarks=self.show_landmarks)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_stats_signal.connect(self.update_stats)
        self.video_thread.start()
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.stats_display.setText("Camera starting..."); self.image_label.setMinimumSize(800,480); self.image_label.setText("")
        self.posture_status.setText("Monitoring Posture..."); self.posture_status.setStyleSheet(posture_style("monitor"))

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread._run_flag = False
            QTimer.singleShot(100, self.finish_video_stop)
        else:
            self.finish_video_stop()

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
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.stats_display.setText("Camera stopped")
        self.posture_status.setText("Camera Stopped"); self.posture_status.setStyleSheet(posture_style("stopped"))
        self.image_label.setText("Click 'Start Camera' to begin webcam feed"); self.image_label.clear()
        QTimer.singleShot(1000, self.check_app_status)

    def check_app_status(self):
        self.stats_display.setText("⏹️ Camera stopped - App is running")

    # events
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
        current_text = self.stats_display.text()
        if "Voice" not in current_text:
            self.stats_display.setText(f"Analysis: {text}")
        if text and not any(w in text.lower() for w in ["detecting","stabilizing","transitioning","confirming","analyzing"]):
            tl = text.lower()
            if "good posture" in tl:
                self.posture_status.setText("Good Posture"); self.posture_status.setStyleSheet(posture_style("good"))
            elif "moderately bad posture" in tl or "moderate" in tl:
                self.posture_status.setText("Moderate Posture Issues"); self.posture_status.setStyleSheet(posture_style("moderate"))
            elif "bad posture" in tl:
                self.posture_status.setText("Poor Posture Detected"); self.posture_status.setStyleSheet(posture_style("bad"))
            elif "no pose" in tl:
                self.posture_status.setText("No Person Detected"); self.posture_status.setStyleSheet(posture_style("stopped"))

    def closeEvent(self, event):
        if self.video_thread.isRunning(): self.video_thread.stop()
        if self.speech_thread.isRunning(): self.speech_thread.stop()
        event.accept()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
