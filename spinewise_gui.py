import os, queue, cv2, csv, json, datetime, time, numpy as np, pandas as pd, mediapipe as mp, threading, backend, \
    speech_recognition as sr
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QStackedWidget, QButtonGroup, QRadioButton, QSizePolicy, QFrame,
    QVBoxLayout, QWidget, QTabWidget, QMainWindow, QFileDialog, QTextEdit, QDoubleSpinBox,
    QScrollArea, QSpinBox, QHBoxLayout, QCheckBox, QFormLayout, QSlider, QGroupBox, QProgressBar,
    QTableWidgetItem, QTableWidget, QGridLayout, QHeaderView, QAction, QLineEdit
)
from PyQt5.QtGui import QImage, QDesktopServices, QPixmap, QFont, QIcon, QFontDatabase, QPalette, QBrush, QPainter, \
    QColor
from PyQt5.QtCore import Qt, QUrl, QSize, QPropertyAnimation, QRect, QEasingCurve, QThread, pyqtSignal, QEvent, QTimer, \
    QObject, QTime

from backend import (
    analyze_posture, get_pose_landmarker, get_face_landmarker, draw_landmarks,
    normalize_lighting, is_calibrating, calibration_start_time, calibration_data,
    set_gui_mode
)
from backend import get_recommendation_context

# theme importing
from spinewise_theme import (
    apply_palette, APP_QSS, RECS_QSS, PRODUCT_CARD_QSS, LIVE_IMAGE_QSS, STATUS_PANEL_QSS,
    BTN_SUCCESS_QSS, POPUP_FRAME_QSS, DOT_QSS, posture_style, voice_status_style
)
from voice_config import voice_config
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
        self.tab_active = False
        self.currentTime = QTime(0, 0, 0, 0)
        self.file_position = 0
        self.frequency = {}
        self.countList = []
        self.prev = ""
        self.substring = ""

        plt.style.use('bmh')
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.ax1 = self.figure.add_subplot(121)
        self.canvas.ax2 = self.figure.add_subplot(122)
        self.xdata = [0]
        self.ydata = [[0], [0]]

        # additional file exception handling for stats.json file
        try:
            with open("stats.json", "r") as json_file:
                data = json.load(json_file)
                self.currentTime = QTime.fromString(data['time'], "hh:mm:ss")
                self.frequency = data['frequency']
        except FileNotFoundError:
            with open("stats.json", "w") as json_file:
                json.dump({"day": datetime.datetime.now().day, "time": "00:00:00", "frequency": {}}, json_file)
            try:
                size = os.path.getsize(self.file_path)
                if len(self.frequency) == 0 and size > 0:
                    with open(self.file_path, "r") as temp:
                        key = (next(iter(temp)))[0:10]
                        self.frequency.update({key: ""})
            except FileNotFoundError as e:
                print(f"[ANALYTICS] Failed to read CSV: {e}")

    def set_tab(self, index):
        self.tab_active = (index == 1)

    def run(self):
        while self.isRunning():
            if self.tab_active:
                self.progress.emit()
                try:
                    self.read_new_data_incrementally()
                    self.most_frequent()
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
                            self.substring = (row[0])[:10]
                            self.ydata[0].append(confidence)
                            self.ydata[1].append(head_tilt)
                            self.xdata.append(self.xdata[-1] + interval)
                            if (self.prev != "" and self.substring != self.prev):
                                self.most_frequent()
                                self.countList.clear()
                            (self.countList).append(row[3])
                            self.prev = self.substring
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

    # responsible for collecting mode of posture score associated with each day
    def most_frequent(self):
        if len(self.countList) > 0:
            self.frequency.update({self.prev: max(set(self.countList), key=self.countList.count)})
        data = {}
        with open("stats.json", "r") as json_file:
            data = json.load(json_file)
        data['frequency'] = self.frequency
        with open("stats.json", "w") as json_file:
            json.dump(data, json_file)

    def stop(self):
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

    def enable_listening(self):
        self.listening_enabled = True

    def disable_listening(self):
        self.listening_enabled = False

    def stop(self):
        self._run_flag = False
        self.listening_enabled = False
        self.wait()


# video thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_stats_signal = pyqtSignal(str)
    update_reco_context_signal = pyqtSignal(dict)

    def __init__(self, show_landmarks=False):
        super().__init__()
        self._run_flag = True
        self.pose_landmarker = None
        self.face_landmarker = None
        self.raw_queue = None
        self.processed_queue = None
        self.show_landmarks = show_landmarks

    def set_landmark_visibility(self, show_landmarks):
        self.show_landmarks = show_landmarks

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
                    if result in ("good", "moderate", "bad"):
                        self.update_stats_signal.emit(result)
                    # send reco context for recommendation tab
                    ctx = get_recommendation_context()
                    self.update_reco_context_signal.emit(ctx)

                else:
                    self.update_stats_signal.emit("No pose detected")
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
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


from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtCore import QByteArray


class ImageLoader(QObject):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            QObject.__init__(cls._instance)
            cls._instance.nam = QNetworkAccessManager()
            cls._instance.cache = {}
        return cls._instance

    def fetch(self, url, on_ready):
        if not url:
            on_ready(None);
            return
        if url in self.cache:
            on_ready(self.cache[url]);
            return
        reply = self.nam.get(QNetworkRequest(QUrl(url)))
        reply.finished.connect(lambda r=reply, cb=on_ready, u=url: self._done(r, cb, u))

    def _done(self, reply, on_ready, url):
        data = reply.readAll()
        pix = QPixmap()
        if not data.isEmpty():
            pix.loadFromData(QByteArray(data))
        self.cache[url] = pix if not pix.isNull() else None
        on_ready(self.cache[url])
        reply.deleteLater()


# product card
class ProductCard(QFrame):
    def __init__(self, title, category, why, price_text, rating=None, reviews=None, url=None, image_url="",
                 parent=None):
        super().__init__(parent)
        self.url = (url or "").strip()
        self.setObjectName("ProductCard")
        self.setStyleSheet(PRODUCT_CARD_QSS)
        v = QVBoxLayout(self)
        self.img = QLabel()
        self.img.setFixedHeight(140)
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setStyleSheet("background:#F6F8FF; border-radius:10px;")
        v.addWidget(self.img)
        title_lbl = QLabel(title or "—")
        title_lbl.setObjectName("CardTitle")
        title_lbl.setWordWrap(True)
        pill = QLabel(category or "—")
        pill.setObjectName("Pill")
        top = QHBoxLayout();
        top.addWidget(title_lbl, 1);
        top.addWidget(pill, 0, Qt.AlignRight);
        v.addLayout(top)
        why_lbl = QLabel(why or "—");
        why_lbl.setObjectName("Why");
        why_lbl.setWordWrap(True);
        v.addWidget(why_lbl)
        meta = [];
        if self.url:
            self.setCursor(Qt.PointingHandCursor)
        if rating: meta.append(f"⭐ {rating}")
        if reviews: meta.append(f"({reviews} reviews)")
        meta_lbl = QLabel(" ".join(meta) if meta else " ");
        meta_lbl.setObjectName("Meta");
        v.addWidget(meta_lbl)
        bottom = QHBoxLayout()
        price_lbl = QLabel(price_text or "—");
        price_lbl.setObjectName("Price");
        bottom.addWidget(price_lbl)
        if url:
            link = QLabel(f'<a href="{url}">Open</a>');
            link.setOpenExternalLinks(True);
            bottom.addStretch(1);
            bottom.addWidget(link)
        v.addLayout(bottom)
        # Load image async
        if image_url:
            ImageLoader().fetch(image_url, self._set_image)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.url:
            QDesktopServices.openUrl(QUrl(self.url))
        super().mousePressEvent(e)

    def _set_image(self, pix):
        if pix:
            self.img.setPixmap(pix.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img.setText("No image")


class CoachCard(QFrame):
    def __init__(self, title, bullets=None, chips=None, confidence=None, parent=None):
        super().__init__(parent)
        self.setObjectName("CoachCard")
        self.setStyleSheet("""
            #CoachCard { background:#FFFFFF; border:1px solid #DFE8FF; border-radius:12px; }
            #Title { font-weight:600; font-size:14px; color:#0F2238; }
            #Chip { padding:2px 8px; border-radius:10px; background:#EAF2FF; color:#0B5CAD; font-size:11px; }
            #Body { color:#2E3C51; }
            #Conf { color:#5A6B84; font-size:11px; }
        """)
        v = QVBoxLayout(self);
        v.setContentsMargins(12, 10, 12, 12);
        v.setSpacing(6)

        t = QLabel(title or "—");
        t.setObjectName("Title")
        v.addWidget(t)

        if chips:
            row = QHBoxLayout()
            for c in chips[:4]:
                chip = QLabel(c);
                chip.setObjectName("Chip")
                row.addWidget(chip)
            row.addStretch(1)
            v.addLayout(row)

        if bullets:
            for b in bullets[:4]:
                bl = QLabel("• " + b);
                bl.setObjectName("Body")
                v.addWidget(bl)

        if confidence is not None:
            v.addSpacing(4)
            v.addWidget(QLabel(f"Confidence: {int(confidence * 100)}%"), 0, Qt.AlignLeft)


# main window
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        set_gui_mode(True)
        self.setWindowTitle("SpineWise Posture App")
        self.setGeometry(100, 100, 1400, 900)

        apply_palette(self)
        self.setStyleSheet(APP_QSS)
        self.app_font = QFont("Segoe UI", 10)

        self.tab_widget = QTabWidget();
        self.tab_widget.setElideMode(Qt.ElideRight)
        main_layout = QVBoxLayout();
        main_layout.setAlignment(Qt.AlignTop);
        main_layout.setContentsMargins(16, 16, 16, 16);
        main_layout.addWidget(self.tab_widget)
        central = QWidget();
        central.setLayout(main_layout);
        self.setCentralWidget(central)

        self.live_tab = QWidget();
        self.analytics_tab = QWidget();
        self.recommendations_tab = QWidget();
        self.settings_tab = QWidget();
        self.about_tab = QWidget()
        self.tab_widget.addTab(self.live_tab, "Dashboard")
        self.tab_widget.addTab(self.analytics_tab, "Analytics")
        self.tab_widget.addTab(self.recommendations_tab, "Recommendations")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.about_tab, "About Us")

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
        self.init_about_tab()

        # recommendation tab
        self.current_reco_context = {"pattern": None, "confidence": None, "tags": [], "evidence": {}}
        self.video_thread.update_reco_context_signal.connect(self._on_reco_context)

        # default engine mode
        # Sets GUI dropdown and backend mode to rule based at startup
        if hasattr(self, "engine_combo"):
            self.engine_combo.setCurrentIndex(0)
            backend.set_detection_mode("rules")
        self.init_recommendations_tab()
        self.speech_thread.start()

        # settings handler for engine switch

    def _on_engine_changed(self, idx):
        mode = self.engine_combo.itemData(idx)
        try:
            backend.set_detection_mode(mode)
            self.set_status_detail(f"Detection engine set to: {self.engine_combo.currentText()}")
        except Exception as e:
            self.set_status_detail(f"Failed to switch engine: {e}")

    def _render_coach_from_context(self, ctx: dict):
        pat = ctx.get("pattern") or "—"
        conf = ctx.get("confidence")
        tags = ", ".join(ctx.get("tags") or []) or "—"
        conf_txt = f"{int(conf * 100)}%" if conf is not None else "—"
        if hasattr(self, "ctx_summary"):
            self.ctx_summary.setText(f"Pattern: {pat}    Confidence: {conf_txt}    Tags: {tags}")

        # clear grid
        while hasattr(self, "coach_grid") and self.coach_grid.count():
            w = self.coach_grid.takeAt(0).widget()
            if w: w.deleteLater()

        # choose up to 3 concise cards
        cards = []
        if pat == "forward_head":
            cards = [
                ("Setup", ["Raise monitor to eye level", "Keep keyboard close", "Sit close to desk"],
                 ["monitor_low", "reach"]),
                ("Exercises", ["Chin tucks 3×10", "Wall slides 2×10", "Thoracic extension 2×10"], ["cervical"]),
                ("Habits", ["20–20–20 eye breaks", "Stand 5 min each hour"], ["phone_neck"])
            ]
        # rounded shoulders isnt currently being used
        elif pat == "rounded_shoulders":
            cards = [
                ("Setup", ["Elbows under shoulders", "Armrests just below elbows"], ["reach"]),
                ("Exercises", ["Doorway pec stretch 3×30s", "Band pull-aparts 3×15"], ["pec_short"]),
                ("Habits", ["Daily posture reset cue", "Light rows 2×/week"], [])
            ]
        elif pat == "slouched_sitting":
            cards = [
                ("Setup", ["Hips ≈ knees", "Feet fully supported", "Small lumbar support"], ["pelvis"]),
                ("Exercises", ["Brugger relief 3×/day", "Glute squeeze 3×10"], []),
                ("Habits", ["Stand for calls", "Micro-break timer"], [])
            ]
        else:
            cards = [("Maintenance", ["Micro-breaks", "5-min mobility daily", "Alternate sit/stand"], [])]

        for i, (title, bullets, chips) in enumerate(cards):
            r, c = divmod(i, 3)
            self.coach_grid.addWidget(CoachCard(title, bullets, chips, conf), r, c)

    def _on_reco_context(self, ctx: dict):
        self.current_reco_context = ctx or {"pattern": None, "confidence": None, "tags": [], "evidence": {}}

        pat = self.current_reco_context.get("pattern") or "—"
        conf = self.current_reco_context.get("confidence")
        tags = ", ".join(self.current_reco_context.get("tags") or []) or "—"

        if hasattr(self, "reco_pattern_lbl"):
            self.reco_pattern_lbl.setText(f"Pattern: {pat}")
        if hasattr(self, "reco_conf_lbl"):
            self.reco_conf_lbl.setText(f"Confidence: {'—' if conf is None else f'{int(conf * 100)}%'}")
        if hasattr(self, "reco_tags_lbl"):
            self.reco_tags_lbl.setText(f"Tags: {tags}")

        # new: refresh coach cards
        self._render_coach_from_context(self.current_reco_context)

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
                {"title": "Memory Foam Neck Pillow", "category": "Neck Pillow",
                 "why": "Supports cervical alignment during long sessions.", "confidence": None, "price_text": "$24.99",
                 "rating": "4.5", "reviews": "12,345", "url": ""},
                {"title": "Adjustable Posture Corrector", "category": "Posture Corrector",
                 "why": "Gently retracts shoulders to counter slouching.", "confidence": None, "price_text": "$29.99",
                 "rating": "4.3", "reviews": "8,901", "url": ""},
                {"title": "Resistance Bands Set", "category": "Resistance Bands",
                 "why": "Helps strengthen scapular stabilizers.", "confidence": None, "price_text": "$19.99",
                 "rating": "4.6", "reviews": "22,101", "url": ""},
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
                price_text=p.get("price_text"), rating=p.get("rating"), reviews=p.get("reviews"), url=p.get("url"),
                image_url=p.get("image_url", "")
            )
            self.product_grid.addWidget(card, r, c)
        self.product_grid.setRowStretch((len(products) + cols - 1) // cols + 1, 1)

    def update_stopwatch(self):
        self.graph_thread.currentTime = self.graph_thread.currentTime.addMSecs(self.timer.interval())

    def init_recommendations_tab(self):
        self.recommendations_tab.setObjectName("RecsTab")
        root = QWidget();
        root.setObjectName("RecsRoot");
        root.setStyleSheet(RECS_QSS)
        outer = QVBoxLayout(root);
        outer.setContentsMargins(16, 16, 16, 16);
        outer.setSpacing(14)

        # live coach might remove kind of trash updates very quickly
        coach_title = QLabel("Live Coach")
        coach_title.setProperty("class", "SectionTitle")
        coach_title.setAlignment(Qt.AlignLeft)
        outer.addWidget(coach_title)

        self.ctx_summary = QLabel("Pattern: —    Confidence: —    Tags: —")
        self.ctx_summary.setStyleSheet("color:#5A6B84; margin-bottom:6px;")
        outer.addWidget(self.ctx_summary)

        self.coach_container = QWidget()
        self.coach_grid = QGridLayout(self.coach_container)
        self.coach_grid.setContentsMargins(0, 0, 0, 0)
        self.coach_grid.setHorizontalSpacing(12)
        self.coach_grid.setVerticalSpacing(12)
        outer.addWidget(self.coach_container)

        # products area
        cards_title = QLabel("Recommended Products")
        cards_title.setProperty("class", "SectionTitle")
        cards_title.setAlignment(Qt.AlignLeft)
        outer.addWidget(cards_title)

        tools = QHBoxLayout()
        refresh_btn = QPushButton("Refresh from CSV")
        refresh_btn.setObjectName("Primary");
        refresh_btn.setProperty("class", "Primary")
        refresh_btn.clicked.connect(self._on_refresh_products_from_csv)
        tools.addWidget(refresh_btn, 0);
        tools.addStretch(1)
        outer.addLayout(tools)

        self.products_container = QWidget()
        self.product_grid = QGridLayout(self.products_container)
        self.product_grid.setContentsMargins(0, 0, 0, 0)
        self.product_grid.setHorizontalSpacing(12)
        self.product_grid.setVerticalSpacing(12)
        outer.addWidget(self.products_container)

        controls_title = QLabel("Fine-tune & Export")
        controls_title.setObjectName("SubTitle");
        controls_title.setProperty("class", "SubTitle")
        outer.addWidget(controls_title)

        filters_row = QHBoxLayout()
        issue_lbl = QLabel("Focus issues:")
        self.issue_filter_input = QLineEdit()
        self.issue_filter_input.setPlaceholderText("e.g., forward head, rounded shoulders")
        filters_row.addWidget(issue_lbl);
        filters_row.addWidget(self.issue_filter_input, 1)
        budget_lbl = QLabel("Budget:")
        self.budget_input = QLineEdit();
        self.budget_input.setPlaceholderText("Max $ (optional)")
        self.budget_input.setFixedWidth(160)
        filters_row.addWidget(budget_lbl);
        filters_row.addWidget(self.budget_input)
        outer.addLayout(filters_row)

        self.recs_table = QTableWidget();
        self.recs_table.setColumnCount(6)
        # more generic headers so tips/products both make sense
        self.recs_table.setHorizontalHeaderLabels(["Item", "Kind", "Details", "Confidence", "Price", "Link"])
        self.recs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        outer.addWidget(self.recs_table)

        buttons = QHBoxLayout()
        generate_btn = QPushButton("Generate Recommendations")
        generate_btn.setObjectName("Primary");
        generate_btn.setProperty("class", "Primary")
        generate_btn.clicked.connect(self._on_generate_recommendations)
        save_btn = QPushButton("Save as CSV");
        save_btn.clicked.connect(self._on_save_recommendations_csv)
        buttons.addWidget(generate_btn);
        buttons.addWidget(save_btn);
        buttons.addStretch(1)
        outer.addLayout(buttons)

        scroll = QScrollArea();
        scroll.setWidgetResizable(True);
        scroll.setWidget(root)
        main_layout = QVBoxLayout();
        main_layout.addWidget(scroll)
        self.recommendations_tab.setLayout(main_layout)

        # initial content
        self._on_refresh_products_from_csv()
        # also render coach once with current (empty) context
        self._render_coach_from_context(self.current_reco_context)

    # live tab
    def init_live_tab(self):
        live_wrapper = QWidget();
        live_wrapper.setAttribute(Qt.WA_StyledBackground, True);
        live_wrapper.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(live_wrapper);
        layout.setSpacing(15)

        title = QLabel("Live Posture Monitoring");
        title.setFont(QFont(self.app_font.family(), 18, QFont.DemiBold));
        title.setAlignment(Qt.AlignCenter);
        title.setStyleSheet("color: #0B5CAD;")
        layout.addWidget(title)

        # controls moved above the camera feed
        controls_layout = QHBoxLayout();
        controls_layout.setSpacing(15)
        icon_size = QSize(30, 30)

        self.start_button = QPushButton("Start Camera");
        self.start_button.setIcon(QIcon("assets/start_icon.png"))
        self.start_button.setIconSize(icon_size);
        self.start_button.setMinimumHeight(44);
        self.start_button.clicked.connect(self.start_video)
        self.start_button.setStyleSheet("""QPushButton {color: black;}""")
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera");
        self.stop_button.setIcon(QIcon("assets/stop_icon.png"))
        self.stop_button.setIconSize(icon_size);
        self.stop_button.setMinimumHeight(44);
        self.stop_button.clicked.connect(self.stop_video);
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""QPushButton {color: black;}""")
        controls_layout.addWidget(self.stop_button)

        self.calibrate_button = QPushButton("Calibrate");
        self.calibrate_button.setIcon(QIcon("assets/calibrate_icon.png"))
        self.calibrate_button.setIconSize(icon_size);
        self.calibrate_button.setMinimumHeight(44);
        self.calibrate_button.setStyleSheet(BTN_SUCCESS_QSS);
        self.calibrate_button.clicked.connect(self.start_calibration)
        controls_layout.addWidget(self.calibrate_button)

        controls_layout.addStretch(1)
        layout.addLayout(controls_layout)

        self.image_label = QLabel("Click 'Start Camera' to begin webcam feed");
        self.image_label.setAlignment(Qt.AlignCenter);
        self.image_label.setMinimumSize(800, 480)
        self.image_label.setStyleSheet(LIVE_IMAGE_QSS);
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        # combined status box
        self.status_box = QLabel()
        self.status_box.setAlignment(Qt.AlignCenter)
        self.status_box.setWordWrap(True)
        self.status_box.setStyleSheet(posture_style("stopped"))
        self.set_status("Posture Status: Not monitoring", "Detailed Status: Click 'Start Camera' to begin monitoring",
                        style_kind="stopped")
        layout.addWidget(self.status_box)

        self.voice_status = QLabel("🎤 Voice Status: Initializing...");
        self.voice_status.setAlignment(Qt.AlignCenter);
        self.voice_status.setStyleSheet(voice_status_style("neutral"))
        layout.addWidget(self.voice_status)

        self.live_tab.setLayout(QVBoxLayout());
        self.live_tab.layout().addWidget(live_wrapper)

    # analytics tab
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

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_stopwatch)
        self.timer.start()

        self.graph_thread.start()

        self.tab_widget.currentChanged.connect(self.graph_thread.set_tab)

        desc = QLabel("This analytics panel will show posture metrics over time.")
        desc.setFont(QFont("Press Start 2P", 9))
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.analytics_tab.setLayout(layout)

    # settings tab
    def init_settings_tab(self):
        layout = QVBoxLayout();
        layout.setSpacing(12)

        visual_group = QGroupBox("Visual Settings");
        v_layout = QFormLayout()
        self.landmark_checkbox = QCheckBox("Show pose landmarks on camera feed");
        self.landmark_checkbox.setChecked(self.show_landmarks);
        self.landmark_checkbox.stateChanged.connect(self.toggle_landmark_visibility)
        v_layout.addRow(QLabel("Landmarks:"), self.landmark_checkbox)
        landmark_info = QLabel("When enabled, shows pose detection points and connections on the video feed");
        landmark_info.setStyleSheet("color: #5A6B84;")
        v_layout.addRow("", landmark_info)
        visual_group.setLayout(v_layout);
        layout.addWidget(visual_group)

        # detection engine group
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
        #

        notif_group = QGroupBox("Notification Settings");
        n_layout = QFormLayout()
        vol_row = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal);
        self.volume_slider.setRange(0, 100);
        self.volume_slider.setValue(self.notification_volume)
        self.volume_slider.setTickPosition(QSlider.TicksBelow);
        self.volume_slider.setTickInterval(10)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_label = QLabel(f"{self.notification_volume}%");
        self.volume_label.setFixedWidth(60)
        vol_row.addWidget(self.volume_slider, 1);
        vol_row.addWidget(self.volume_label, 0, Qt.AlignRight)
        n_layout.addRow(QLabel("Notification Volume:"), vol_row)

        self.beep_interval_spinbox = QDoubleSpinBox();
        self.beep_interval_spinbox.setRange(0.5, 10.0);
        self.beep_interval_spinbox.setSingleStep(0.5)
        self.beep_interval_spinbox.setValue(self.beep_interval);
        self.beep_interval_spinbox.setSuffix(" seconds");
        self.beep_interval_spinbox.valueChanged.connect(self._on_beep_interval_changed)
        n_layout.addRow(QLabel("Beep Interval:"), self.beep_interval_spinbox)

        self.alert_duration_spinbox = QSpinBox();
        self.alert_duration_spinbox.setRange(1, 60);
        self.alert_duration_spinbox.setValue(int(self.alert_duration))
        self.alert_duration_spinbox.setSuffix(" seconds");
        self.alert_duration_spinbox.valueChanged.connect(self._on_alert_duration_changed)
        n_layout.addRow(QLabel("Alert Duration:"), self.alert_duration_spinbox)

        notif_group.setLayout(n_layout);
        layout.addWidget(notif_group)

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

        # Voice Commands Customization Group
        commands_group = QGroupBox("Voice Command Bindings")
        commands_layout = QFormLayout()

        # Info label
        cmd_info = QLabel("Customize trigger words for each command (comma-separated)")
        cmd_info.setWordWrap(True)
        cmd_info.setStyleSheet("color: #5A6B84; font-size: 10px; margin-bottom: 10px;")
        commands_layout.addRow(cmd_info)

        # Create text fields for each command type
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
            # Get current triggers and display them
            current_triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(current_triggers))
            input_field.setPlaceholderText(f"Enter trigger words for {display_name}")

            # Connect to update function
            input_field.editingFinished.connect(
                lambda t=cmd_type, f=input_field: self._on_command_triggers_changed(t, f.text())
            )

            self.command_inputs[cmd_type] = input_field
            commands_layout.addRow(f"{display_name}:", input_field)

        # Reset to defaults button
        reset_commands_btn = QPushButton("Reset Commands to Defaults")
        reset_commands_btn.clicked.connect(self._reset_voice_commands)
        reset_commands_btn.setStyleSheet("padding: 6px 12px; margin-top: 10px;")
        commands_layout.addRow("", reset_commands_btn)

        commands_group.setLayout(commands_layout)
        layout.addWidget(commands_group)

        # Voice Help Section (enhanced)
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

        export_button = QPushButton("💾 Export Log as CSV")
        export_button.clicked.connect(self.export_log)
        export_button.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(export_button)

        clear_log_button = QPushButton("🗑️ Clear Log Data")
        clear_log_button.clicked.connect(self.clear_log)
        clear_log_button.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        data_btn_layout.addWidget(clear_log_button)

        # data_layout.addRow("Actions:", data_btn_layout)

        # data_info = QLabel("Export your posture data to CSV format or clear all logged data")
        # data_info.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        # data_layout.addRow("", data_info)

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

        settings_inner = QWidget();
        settings_inner.setLayout(layout)
        settings_scroll = QScrollArea();
        settings_scroll.setWidgetResizable(True);
        settings_scroll.setWidget(settings_inner)
        outer = QVBoxLayout();
        outer.addWidget(settings_scroll);
        self.settings_tab.setLayout(outer)

    def init_about_tab(self):
        container = QWidget()
        container.setStyleSheet("background-color: #EAF2FF;")
        outer = QVBoxLayout(container);
        outer.setContentsMargins(20, 20, 20, 20);
        outer.setSpacing(14)

        title = QLabel("Meet the Devs of Spinewise");
        title.setFont(QFont(self.app_font.family(), 18, QFont.DemiBold));
        title.setAlignment(Qt.AlignCenter);
        title.setStyleSheet("color: #0B5CAD;")
        outer.addWidget(title)

        self.carousel_widget = QStackedWidget();
        self.carousel_widget.setMinimumHeight(int(self.height() * 0.55))
        outer.addWidget(self.carousel_widget)

        devs_info = [
            ("Emdya Permuy-Llovio ", "Product Manager", "assets/dev1.png", "https://www.linkedin.com/in/emdyapermuy/",
             "https://github.com/Emdya"),
            ("Juan Mieses", "Fullstack Development ", "assets/dev2.png", "https://www.linkedin.com/in/juanmieses003/",
             "https://github.com/Jmies-27"),
            ("Javier Brasil", "Fullstack Development", "assets/dev3.png",
             "https://www.linkedin.com/in/javier-a-brasil/", "https://github.com/SlyyJavii"),
            ("John Pena ", "Backend and Machine Learning Development", "assets/dev4.png",
             "https://www.linkedin.com/in/johnpenacs/", "https://github.com/jpena173"),
            ("Jake Rodriguez", "Visual and Audio Alert System", "assets/dev5.png",
             "https://www.linkedin.com/in/jake-rodriguez-917a24142/", "https://github.com/jrodr995"),
            ("Oleh Krainyk", "UI Development", "assets/dev6.jpg", "https://www.linkedin.com/in/oleh-krainyk/",
             "https://github.com/olegKrainyk")
        ]
        captions = [
            "Emdya Permuy-Llovio is an Undergraduate BS in Computer Science student at Florida International University... ",
            "Juan A. Mieses is a Florida International University Undergraduate student pursuing a Bachelor's Degree... ",
            "Javier builds solid infrastructure and efficient code.",
            "John Pena is an aspiring undergraduate studying Computer Science, preferring cybersecurity tasks and backend development...",
            "Jake crafts beautiful alert systems and UI animations.",
            "Oleh Krainyk is a UI developer with a passion for creating intuitive and engaging user experiences."
        ]

        for idx, (name, role, img_path, linkedin, github) in enumerate(devs_info):
            card = QWidget();
            card_layout = QVBoxLayout(card);
            card_layout.setContentsMargins(30, 10, 30, 10);
            card_layout.setSpacing(12);
            card_layout.setAlignment(Qt.AlignCenter)
            card.setStyleSheet("background-color: #FFFFFF; border: 0; border-radius: 0px;")
            image = QLabel();
            image.setPixmap(QPixmap(img_path).scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation));
            image.setAlignment(Qt.AlignCenter);
            card_layout.addWidget(image)
            name_label = QLabel(name);
            name_label.setFont(QFont(self.app_font.family(), 12, QFont.Medium));
            name_label.setAlignment(Qt.AlignCenter);
            name_label.setStyleSheet("color: #0F2238; padding: 8px; font-size: 14px;");
            card_layout.addWidget(name_label)

            role_row = QHBoxLayout();
            role_row.setAlignment(Qt.AlignCenter)
            github_button = QPushButton();
            github_button.setCursor(Qt.PointingHandCursor);
            github_button.setIcon(QIcon("assets/icons/GitHub_Invertocat_Dark.png"))
            github_button.setIconSize(QSize(20, 20));
            github_button.setFixedSize(28, 28);
            github_button.setStyleSheet(
                "QPushButton { background: transparent; border: none; } QPushButton:hover { background: #F0F4FF; }")
            github_button.clicked.connect(lambda _, url=github: QDesktopServices.openUrl(QUrl(url)));
            role_row.addWidget(github_button)

            role_label = QLabel(role);
            role_label.setStyleSheet("color: #5A6B84;");
            role_row.addWidget(role_label)

            linkedin_button = QPushButton();
            linkedin_button.setCursor(Qt.PointingHandCursor);
            linkedin_button.setIcon(QIcon("assets/icons/LinkedIn_logo_initials.png"))
            linkedin_button.setIconSize(QSize(20, 20));
            linkedin_button.setFixedSize(28, 28);
            linkedin_button.setStyleSheet(
                "QPushButton { background: transparent; border: none; } QPushButton:hover { background: #F0F4FF; }")
            linkedin_button.clicked.connect(lambda _, url=linkedin: QDesktopServices.openUrl(QUrl(url)));
            role_row.addWidget(linkedin_button)

            card_layout.addLayout(role_row)

            caption_label = QLabel(captions[idx]);
            caption_label.setWordWrap(True);
            caption_label.setAlignment(Qt.AlignCenter);
            caption_label.setStyleSheet("color: #2E3C51;");
            card_layout.addWidget(caption_label)
            self.carousel_widget.addWidget(card)

        nav_layout = QHBoxLayout();
        nav_layout.setAlignment(Qt.AlignCenter)
        left_btn = QPushButton("<");
        left_btn.setFixedSize(36, 36);
        left_btn.setStyleSheet(
            "QPushButton { background: #EAF2FF; color: #0F2238; border-radius: 6px; padding: 5px; } QPushButton:hover { background: #DDEBFF; }")
        left_btn.clicked.connect(lambda: self.carousel_widget.setCurrentIndex(
            (self.carousel_widget.currentIndex() - 1) % self.carousel_widget.count()))
        nav_layout.addWidget(left_btn)

        dot_group = QButtonGroup();
        self.pagination_dots = []
        for i in range(self.carousel_widget.count()):
            dot = QRadioButton();
            dot.setStyleSheet(DOT_QSS)
            dot.toggled.connect(lambda checked, idx=i: self.carousel_widget.setCurrentIndex(idx) if checked else None)
            dot_group.addButton(dot);
            self.pagination_dots.append(dot);
            nav_layout.addWidget(dot)

        right_btn = QPushButton(">");
        right_btn.setFixedSize(36, 36);
        right_btn.setStyleSheet(
            "QPushButton { background: #EAF2FF; color: #0F2238; border-radius: 6px; padding: 5px; } QPushButton:hover { background: #DDEBFF; }")
        right_btn.clicked.connect(lambda: self.carousel_widget.setCurrentIndex(
            (self.carousel_widget.currentIndex() + 1) % self.carousel_widget.count()))
        nav_layout.addWidget(right_btn)
        outer.addLayout(nav_layout)

        if self.pagination_dots: self.pagination_dots[0].setChecked(True)

        def sync_dots(index):
            if 0 <= index < len(self.pagination_dots): self.pagination_dots[index].setChecked(True)

        self.carousel_widget.currentChanged.connect(sync_dots)

        scroll = QScrollArea();
        scroll.setWidgetResizable(True);
        scroll.setWidget(container)
        layout = QVBoxLayout();
        layout.addWidget(scroll);
        self.about_tab.setLayout(layout)

    # main code when clicking generate recommendation
    def _on_generate_recommendations(self):
        if getattr(self, "_recs_busy", False):
            return
        self._recs_busy = True
        self.set_status_detail("Finding products…")
        QTimer.singleShot(0, self._generate_recommendations_inner)

    # recs logic
    def _generate_recommendations_inner(self):
        try:
            # Live context
            ctx = getattr(self, "current_reco_context",
                          {"pattern": None, "confidence": None, "tags": [], "evidence": {}})
            pat = ctx.get("pattern")
            conf = ctx.get("confidence")

            # Context aware coach tip always prepended so the table isn't empty
            if pat == "forward_head":
                plan_row = [{
                    "title": "Coach tip: Counter forward head",
                    "category": "Plan",
                    "why": "Raise monitor to eye level, 20–20–20 breaks, chin tucks (3×10).",
                    "confidence": conf, "price_text": "—", "url": ""
                }]
            elif pat == "slouched_sitting":
                plan_row = [{
                    "title": "Coach tip: Neutral pelvis + foot support",
                    "category": "Plan",
                    "why": "Seat height hips≈knees, feet fully supported, slight lumbar support.",
                    "confidence": conf, "price_text": "—", "url": ""
                }]
            else:
                plan_row = [{
                    "title": "Coach tip: Maintain neutral",
                    "category": "Plan",
                    "why": "Short posture-check timer + micro-breaks to keep consistency.",
                    "confidence": conf, "price_text": "—", "url": ""
                }]

            # product lookup
            references = getattr(backend, "get_recommendation_references", lambda: {})()

            focus_text = (self.issue_filter_input.text() or "").strip()
            extra_focus = [s.strip() for s in focus_text.split(",") if s.strip()]

            budget_raw = (self.budget_input.text() or "").strip()
            try:
                budget = float(budget_raw) if budget_raw else None
            except ValueError:
                budget = None

            # build issues user focus > live pattern
            def _norm(s):
                return s.lower().replace(" ", "_")

            KNOWN = {"forward_head", "slouched_sitting"}  # shoulders removed for now

            issues = [_norm(s) for s in (extra_focus or []) if _norm(s) in KNOWN]

            # only trust live pattern if confident enough
            if not issues and pat in KNOWN and (conf is None or conf >= 0.5):
                issues = [pat]

            # if still nothing, we pass [] to backend it will use mixed base-case queries
            if not issues:
                issues = []

            results = getattr(backend, "query_products_via_serpapi", lambda *a, **k: [])(
                issues=issues,
                references=references,
                extra_focus=extra_focus,
                budget=budget,
                weights=None
            )

            # render
            products = plan_row + (results or [])
            if results:
                self._last_products = products
            else:
                products = getattr(self, "_last_products", products)
            self._display_product_cards(products)
            self._populate_recs_table(products)
            self.set_status_detail("Recommendations updated.")
        except Exception as e:
            self.set_status_detail(f"Failed to get recommendations: {e}")
        finally:
            self._recs_busy = False

    def _populate_recs_table(self, items):
        self.recs_table.setRowCount(0)

        only_plan = bool(items) and all((i.get("category") == "Plan") for i in items)
        self.recs_table.setVisible(not only_plan)
        if only_plan or not items:
            return

        self.recs_table.setRowCount(len(items))
        for r, p in enumerate(items):
            title = p.get("title", "—")
            kind = p.get("category", "—")
            why = p.get("why", "—")
            conf = p.get("confidence", None)
            price = p.get("price_text", "—")
            url = p.get("url", "")

            self.recs_table.setItem(r, 0, QTableWidgetItem(title))
            self.recs_table.setItem(r, 1, QTableWidgetItem(kind))
            self.recs_table.setItem(r, 2, QTableWidgetItem(why))
            conf_display = "—" if conf is None else f"{round(float(conf) * 100):d}%"
            self.recs_table.setItem(r, 3, QTableWidgetItem(conf_display))
            self.recs_table.setItem(r, 4, QTableWidgetItem(price))
            self.recs_table.setItem(r, 5, QTableWidgetItem(url if url else "—"))

    def _on_save_recommendations_csv(self):
        try:
            dest, _ = QFileDialog.getSaveFileName(self, "Save Recommendations", "recommendations.csv",
                                                  "CSV Files (*.csv)")
            if not dest: return
            import csv
            cols = [self.recs_table.horizontalHeaderItem(i).text() for i in range(self.recs_table.columnCount())]
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f);
                writer.writerow(cols)
                for r in range(self.recs_table.rowCount()):
                    row = []
                    for c in range(self.recs_table.columnCount()):
                        item = self.recs_table.item(r, c);
                        row.append(item.text() if item else "")
                    writer.writerow(row)
            self.set_status_detail("Saved recommendations to CSV.")
        except Exception as e:
            self.set_status_detail(f"Save failed: {e}")

    # settings handlers
    def toggle_landmark_visibility(self, state):
        self.show_landmarks = (state == Qt.Checked)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_landmark_visibility(self.show_landmarks)
        self.set_status_detail("Landmarks enabled" if self.show_landmarks else "Landmarks disabled")

    def _on_volume_changed(self, value):
        self.notification_volume = value;
        self.volume_label.setText(f"{value}%");
        backend.update_notification_volume(value)

    def _on_beep_interval_changed(self, value):
        self.beep_interval = value;
        backend.update_beep_interval(value)

    def _on_alert_duration_changed(self, value):
        self.alert_duration = value;
        backend.update_alert_duration(value)

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

        # Use voice_config to match commands
        matched_command = voice_config.match_command(command)

        if matched_command == "calibrate":
            self.start_calibration()
            self.set_status_detail("Voice: Starting calibration...")
        elif matched_command == "stop":
            if self.video_thread.isRunning():
                self.stop_video()
                self.set_status_detail("Voice: Camera stopped. Say 'start' to resume.")
            else:
                self.set_status_detail("Voice: Camera already stopped.")
        elif matched_command == "exit":
            self.set_status_detail("Voice: Exiting...")
            QTimer.singleShot(1000, self.close)
        elif matched_command == "start":
            if not self.video_thread.isRunning():
                self.start_video()
                self.set_status_detail("Voice: Camera started.")
            else:
                self.set_status_detail("Voice: Camera already running.")
        elif any(word in ["help", "commands", "what", "options"] for word in words):
            self.set_status_detail("Voice: Commands: start, stop, cal, exit")
        else:
            self.set_status_detail(f"Voice: Unknown '{command}'")

        QTimer.singleShot(4000, self.reset_voice_status)

    def update_speech_status(self, status):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText(f"🎤 Voice Status: {status}")

    def reset_voice_status(self):
        if hasattr(self, "voice_checkbox") and self.voice_checkbox.isChecked():
            self.voice_status.setText("🎤 Voice Status: Listening...")
            self.voice_status.setStyleSheet(voice_status_style("on"))

    def add_notification(self, text: str):
        """Add and show notification in the popup; update the button badge text."""
        # prepend to internal list
        self.notifications.insert(0, text)
        # add to popup list
        self.notif_popup.add_notification(text)
        # update badge text
        self._update_notif_button_text()

    def _update_notif_button_text(self):
        """Set button text to include unread count (simple badge)."""
        n = len(self.notifications)
        if n == 0:
            self.notif_btn.setText("Notifications")
        else:
            # show count next to label
            self.notif_btn.setText(f"Notifications ({n})")

    def clear_notifications(self):
        """Clear the notification list and restore placeholder."""
        self.notif_menu.clear()
        placeholder = QAction("No notifications", self)
        placeholder.setEnabled(False)
        self.notif_menu.addAction(placeholder)

    def _on_notification_clicked(self, text: str):
        """Handle notification click - can open product recommendations or mark as read."""
        print(f"[NOTIF] clicked: {text}")
        # Example: remove the clicked action (mark as read)
        for act in list(self.notif_menu.actions()):
            if act.text() == text:
                self.notif_menu.removeAction(act)
                break

    def _on_language_changed(self, index):
        """Handle language selection change"""
        language_code = self.language_combo.itemData(index)
        if voice_config.set_language(language_code):
            language_name = self.language_combo.itemText(index)
            self.set_status_detail(f"Language changed to {language_name}")
            # Restart speech thread with new language if it's running
            if self.speech_thread.isRunning() and self.voice_checkbox.isChecked():
                self.speech_thread.disable_listening()
                QTimer.singleShot(500, lambda: self.speech_thread.enable_listening())
        else:
            self.set_status_detail("Failed to change language")

    def _on_command_triggers_changed(self, command_type, text):
        """Handle command trigger text change"""
        triggers = [t.strip() for t in text.split(',') if t.strip()]
        if triggers:
            voice_config.set_command_triggers(command_type, triggers)
            self.set_status_detail(f"Updated {command_type} triggers")

    def _reset_voice_commands(self):
        """Reset all voice commands to defaults"""
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            'Reset Voice Commands',
            'Reset all voice commands to default values?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            voice_config.config["commands"] = voice_config.DEFAULT_CONFIG["commands"].copy()
            voice_config.save_config()

        # Update UI
        for cmd_type, input_field in self.command_inputs.items():
            triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(triggers))

        self.set_status_detail("Voice commands reset to defaults")

    def _generate_voice_help_text(self):
        """Generate dynamic help text based on current voice configuration"""
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
                # Show first few triggers as examples
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
        """Export voice settings to a file"""
        from PyQt5.QtWidgets import QFileDialog

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
                self.set_status_detail(f"Voice settings exported successfully")
            except Exception as e:
                self.set_status_detail(f"Export failed: {e}")

    def _import_voice_settings(self):
        """Import voice settings from a file"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

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

                # Validate imported config has required fields
                if "commands" in imported_config and "language" in imported_config:
                    voice_config.config.update(imported_config)
                    voice_config.save_config()

                    # Update UI to reflect imported settings
                    self._refresh_voice_settings_ui()

                    self.set_status_detail("Voice settings imported successfully")
                else:
                    QMessageBox.warning(self, "Import Error", "Invalid voice settings file")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import: {e}")

    def _refresh_voice_settings_ui(self):
        """Refresh the voice settings UI after import"""
        # Update language combo
        current_lang = voice_config.get_language()
        index = self.language_combo.findData(current_lang)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        # Update command inputs
        for cmd_type, input_field in self.command_inputs.items():
            triggers = voice_config.get_command_triggers(cmd_type)
            input_field.setText(", ".join(triggers))

        # Update help text if visible
        if hasattr(self, 'voice_help_text'):
            self.voice_help_text.setPlainText(self._generate_voice_help_text())

    # calibration
    def start_calibration(self):
        backend.calibration_start_time = time.time()
        backend.is_calibrating = True
        backend.calibration_data = {k: [] for k in backend.calibration_data}
        self.set_status_detail("Calibration started. Hold posture 8s...")

    # logs
    def load_log(self):
        try:
            log_path = os.path.join(os.path.dirname(__file__), "posture_trend_log.csv")
            if os.path.exists(log_path):
                expected = ["Timestamp", "Mode", "Facing", "Posture Status", "Head Tilt", "Confidence Score"]
                df = pd.read_csv(log_path, header=None, names=expected)
                display_df = df[expected].tail(50).reset_index(drop=True)
                self.log_table.setColumnCount(len(expected))
                self.log_table.setHorizontalHeaderLabels(expected)
                self.log_table.setRowCount(len(display_df))
                for r, row in display_df.iterrows():
                    for c, name in enumerate(expected):
                        v = row[name];
                        v = "—" if pd.isna(v) else v
                        item = QTableWidgetItem(str(v))
                        if name == "Posture Status":
                            label = str(v).lower()
                            if "bad" in label:
                                item.setForeground(QColor("#C62828"))
                            elif "good" in label:
                                item.setForeground(QColor("#2E7D32"))
                            elif "moderate" in label:
                                item.setForeground(QColor("#B26A00"))
                        self.log_table.setItem(r, c, item)
            else:
                self.log_table.setRowCount(1);
                self.log_table.setColumnCount(1);
                self.log_table.setItem(0, 0, QTableWidgetItem("📂 No log file found."))
        except Exception as e:
            self.log_table.setRowCount(1);
            self.log_table.setColumnCount(1);
            self.log_table.setItem(0, 0, QTableWidgetItem(f"❌ Error loading log: {e}"))

    # data mgmt
    def clear_log(self):
        try:
            if os.path.exists("posture_trend_log.csv"):
                os.remove("posture_trend_log.csv");
                self.log_table.setRowCount(1);
                self.log_table.setItem(0, 0, QTableWidgetItem("🗑️ Log data cleared."))
            else:
                self.log_table.setRowCount(1);
                self.log_table.setItem(0, 0, QTableWidgetItem("📂 No log file to clear."))
        except Exception as e:
            print("[ERROR] clear_log:", e)

    def export_log(self):
        if os.path.exists("posture_trend_log.csv"):
            dest, _ = QFileDialog.getSaveFileName(self, "Save Log", "posture_trend_log.csv", "CSV Files (*.csv)")
            if dest:
                try:
                    with open("posture_trend_log.csv", "r") as src, open(dest, "w") as dst:
                        dst.write(src.read())
                    self.set_status_detail("Log exported.")
                except Exception as e:
                    self.set_status_detail(f"Export failed: {e}")
        else:
            self.set_status_detail("No log file to export")

    # camera control
    def start_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread._run_flag = False;
            self.video_thread.wait(1000)
        self.video_thread = VideoThread(show_landmarks=self.show_landmarks)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_stats_signal.connect(self.update_stats)
        self.video_thread.update_reco_context_signal.connect(self._on_reco_context)
        self.video_thread.start()
        self.start_button.setEnabled(False);
        self.stop_button.setEnabled(True)
        self.image_label.setMinimumSize(800, 480);
        self.image_label.setText("")
        self.set_status("Monitoring Posture...", "Camera starting...", style_kind="monitor")

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
            try:
                self.video_thread.terminate()
            except Exception:
                pass
        self.update_ui_after_stop()

    def update_ui_after_stop(self):
        self.start_button.setEnabled(True);
        self.stop_button.setEnabled(False)
        self.image_label.setText("Click 'Start Camera' to begin webcam feed");
        self.image_label.clear()
        self.set_status("Camera Stopped", "⏹️ Camera stopped - App is running", style_kind="stopped")

    def check_app_status(self):
        self.set_status_detail("⏹️ Camera stopped - App is running")

    # status helpers
    def set_status(self, headline, detail, style_kind=None):
        self._status_headline = headline
        if style_kind:
            self._status_kind = style_kind
            self.status_box.setStyleSheet(posture_style(style_kind))
        self.status_box.setText(f"<b>{headline}</b><br>{detail}")

    def set_status_detail(self, detail):
        self.status_box.setText(f"<b>{self._status_headline}</b><br>{detail}")

    # events
    def eventFilter(self, source, event):
        return super().eventFilter(source, event)

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def update_stats(self, text):
        if text and not any(
                w in text.lower() for w in ["detecting", "stabilizing", "transitioning", "confirming", "analyzing"]):
            tl = text.lower()
            if "good posture" in tl:
                self.set_status("Good Posture", f"Analysis: {text}", style_kind="good")
            elif "moderately bad posture" in tl or "moderate" in tl:
                self.set_status("Moderate Posture Issues", f"Analysis: {text}", style_kind="moderate")
            elif "bad posture" in tl:
                self.set_status("Poor Posture Detected", f"Analysis: {text}", style_kind="bad")
            elif "no pose" in tl:
                self.set_status("No Person Detected", f"Analysis: {text}", style_kind="stopped")
            else:
                self.set_status_detail(f"Analysis: {text}")
        else:
            self.set_status_detail(f"Analysis: {text}")

    def closeEvent(self, event):

        if self.video_thread.isRunning(): self.video_thread.stop()
        if self.speech_thread.isRunning(): self.speech_thread.stop()

        # responsible for "appending" to json file
        # checks if current day is the same, if so, still copy session time
        # if new day, start from scratch
        data = {}
        timestamp = "00:00:00"
        with open("stats.json", "r") as json_file:
            data = json.load(json_file)
        if data['day'] == datetime.datetime.now().day:
            timestamp = self.graph_thread.currentTime.toString("hh:mm:ss")
        data['time'] = timestamp
        with open("stats.json", "w") as json_file:
            json.dump(data, json_file)

        if self.graph_thread.isRunning(): self.graph_thread.stop()
        event.accept()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
