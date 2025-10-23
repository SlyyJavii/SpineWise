from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QScrollArea, QGraphicsDropShadowEffect, QApplication
)
from PyQt5.QtCore import (
    Qt, QPropertyAnimation, QRect, QEasingCurve,
    pyqtSignal, QTimer, QSize, QPoint
)
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPainterPath,
    QPen, QPixmap
)
from datetime import datetime
from collections import deque


class NotificationBadge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.count = 0
        self.setFixedSize(20, 20)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()

    def setCount(self, count):
        self.count = count
        if count > 0:
            self.show()
            self.update()
        else:
            self.hide()

    def paintEvent(self, event):
        if self.count <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor("#FF4444"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect())

        painter.setPen(QColor("white"))
        font = QFont("Segoe UI", 9 if self.count < 100 else 8, QFont.Bold)
        painter.setFont(font)

        text = str(self.count) if self.count < 100 else "99+"
        painter.drawText(self.rect(), Qt.AlignCenter, text)


class NotificationItem(QFrame):
    clicked = pyqtSignal(str)
    remove_requested = pyqtSignal()

    def __init__(self, text, timestamp, notification_type="info", parent=None):
        super().__init__(parent)
        self.text = text
        self.timestamp = timestamp
        self.notification_type = notification_type
        self.is_read = False
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            NotificationItem {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px;
            }
            NotificationItem:hover {
                background-color: #F8F9FA;
                border: 1px solid #0B5CAD;
            }
        """)

        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)


        icon_label = QLabel()
        icon_map = {
            "bad_posture": "!",
            "good_posture": ":D",
            "calibration": ":0",
            "recommendation": "!!",
            "info": "(i)",
            "error": "!X!"
        }
        icon_label.setText(icon_map.get(self.notification_type, "📢"))
        icon_label.setStyleSheet("font-size: 18px;")
        icon_label.setAlignment(Qt.AlignTop)
        layout.addWidget(icon_label)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        message_label = QLabel(self.text)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        content_layout.addWidget(message_label)

        time_label = QLabel(self.format_timestamp())
        time_label.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 10px;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        content_layout.addWidget(time_label)
        content_layout.addStretch()

        layout.addLayout(content_layout, 1)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #BDC3C7;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #FF5252;
                color: white;
            }
        """)
        close_btn.clicked.connect(self.remove_requested.emit)
        layout.addWidget(close_btn, 0, Qt.AlignTop)

        self.setCursor(Qt.PointingHandCursor)

    def format_timestamp(self):
        now = datetime.now()
        diff = now - self.timestamp

        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        elif diff.days == 0:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.days == 1:
            return "Yesterday"
        else:
            return self.timestamp.strftime("%b %d, %Y")

    def mark_as_read(self):
        self.is_read = True
        self.setStyleSheet("""
            NotificationItem {
                background-color: #F8F9FA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px;
            }
            NotificationItem:hover {
                background-color: #F0F2F5;
                border: 1px solid #8FA3B8;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.text)
            self.mark_as_read()
        super().mousePressEvent(event)


class NotificationDropdown(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initially did NOT have this, but this helps for memory
        # Keeps only the first 15 notifications. VERY LOW
        # I'm scared of putting it any higher
        self.notifications = deque(maxlen=15)
        self.notification_items = []
        self.unread_count = 0
        self.setup_ui()
        self.setVisible(False)

    def setup_ui(self):
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            NotificationDropdown {
                background-color: #FFFFFF;
                border: 1px solid #D5D5D5;
                border-radius: 12px;
                padding: 0px;
            }
        """)

        # Shadow effect. Might comment it out but it looks slick
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #0B5CAD, stop: 1 #084785);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 12px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)

        title = QLabel("Notifications")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Mark all as read button. Doesn't work just yet lol
        self.mark_all_btn = QPushButton("Mark All as Read")
        self.mark_all_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        self.mark_all_btn.clicked.connect(self.mark_all_as_read)
        header_layout.addWidget(self.mark_all_btn)

        layout.addWidget(header_frame)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #FAFBFC;
            }
            QScrollBar:vertical {
                background-color: #F5F5F5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #C0C0C0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #A0A0A0;
            }
        """)

        self.notifications_container = QWidget()
        self.notifications_layout = QVBoxLayout(self.notifications_container)
        self.notifications_layout.setContentsMargins(8, 8, 8, 8)
        self.notifications_layout.setSpacing(6)

        # Empty state label
        self.empty_label = QLabel("No notifications yet")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
                padding: 40px;
            }
        """)
        self.notifications_layout.addWidget(self.empty_label)
        self.notifications_layout.addStretch()

        self.scroll_area.setWidget(self.notifications_container)
        layout.addWidget(self.scroll_area)

        # Footer with clear all
        footer_frame = QFrame()
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border-top: 1px solid #E0E0E0;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
                padding: 8px;
            }
        """)
        footer_layout = QHBoxLayout(footer_frame)

        self.clear_all_btn = QPushButton("🗑️ Clear All")
        self.clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #DC3545;
                border: none;
                font-size: 12px;
                font-family: 'Segoe UI', sans-serif;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #FFF5F5;
                border-radius: 4px;
            }
        """)
        self.clear_all_btn.clicked.connect(self.clear_all_notifications)
        footer_layout.addWidget(self.clear_all_btn)

        footer_layout.addStretch()

        self.count_label = QLabel("0 unread")
        self.count_label.setStyleSheet("""
            QLabel {
                color: #6C757D;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        footer_layout.addWidget(self.count_label)

        layout.addWidget(footer_frame)

        self.setFixedSize(320, 400)

    def add_notification(self, text, notification_type="info"):
        timestamp = datetime.now()

        if self.empty_label.parent():
            self.empty_label.setParent(None)

        # Clear stretch items
        while self.notifications_layout.count() > 0:
            item = self.notifications_layout.itemAt(self.notifications_layout.count() - 1)
            if item.spacerItem():
                self.notifications_layout.removeItem(item)
            else:
                break

        # Create notification item
        item = NotificationItem(text, timestamp, notification_type)
        item.remove_requested.connect(lambda: self.remove_notification(item))

        # Add to top of layout
        self.notifications_layout.insertWidget(0, item)
        self.notification_items.insert(0, item)
        self.notifications.appendleft((text, timestamp, notification_type))

        # Update unread count only if item is not read
        if not item.is_read:
            self.unread_count += 1
            self.update_count_label()

        # Limit visible notifications
        while len(self.notification_items) > 50:
            old_item = self.notification_items.pop()
            old_item.deleteLater()

    def remove_notification(self, item):
        try:
            if item in self.notification_items:
                self.notification_items.remove(item)

            if not item.is_read:
                self.unread_count = max(0, self.unread_count - 1)
                self.update_count_label()

            item.deleteLater()

            if len(self.notification_items) == 0:
                QTimer.singleShot(200, self.show_empty_state)
        except:
            pass

    def show_empty_state(self):
        if len(self.notification_items) == 0:
            self.notifications_layout.addWidget(self.empty_label)
            self.notifications_layout.addStretch()

    def mark_all_as_read(self):
        for item in self.notification_items:
            if not item.is_read:
                item.mark_as_read()
        self.unread_count = 0
        self.update_count_label()

    def clear_all_notifications(self):
        for item in list(self.notification_items):
            item.deleteLater()

        self.notification_items.clear()
        self.notifications.clear()
        self.unread_count = 0
        self.update_count_label()
        self.show_empty_state()

    def update_count_label(self):
        text = f"{self.unread_count} unread" if self.unread_count != 1 else "1 unread"
        self.count_label.setText(text)

    def get_unread_count(self):
        return self.unread_count


class NotificationButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.unread_count = 0
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            NotificationButton {
                background-color: transparent;
                border: 1px solid #D5D5D5;
                border-radius: 20px;
            }
            NotificationButton:hover {
                background-color: #F0F4F8;
                border: 1px solid #0B5CAD;
            }
            NotificationButton:pressed {
                background-color: #E1E8ED;
            }
        """)

        # Create badge
        self.badge = NotificationBadge(self)
        self.badge.move(25, 5)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw bell icon
        painter.setPen(QPen(QColor("#0B5CAD"), 2))

        # Bell shape path
        path = QPainterPath()
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Bell body
        path.moveTo(center_x - 8, center_y + 2)
        path.quadTo(center_x - 8, center_y - 6, center_x, center_y - 8)
        path.quadTo(center_x + 8, center_y - 6, center_x + 8, center_y + 2)
        path.lineTo(center_x - 8, center_y + 2)

        painter.drawPath(path)

        # Bell bottom
        painter.drawLine(center_x - 10, center_y + 2, center_x + 10, center_y + 2)

        # Clapper
        painter.drawEllipse(center_x - 2, center_y + 4, 4, 4)

    def set_unread_count(self, count):
        self.unread_count = count
        self.badge.setCount(count)