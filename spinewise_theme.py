from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

# theme variables
COLORS = {
    "bg_base": "#F5F9FF",
    "bg_white": "#FFFFFF",
    "bg_alt": "#F2F7FF",
    "text_primary": "#0F2238",
    "text_muted": "#2E3C51",
    "brand": "#0B5CAD",
    "brand_hover": "#0D6EDB",
    "border": "#D5E3F4",
    "border_soft": "#C9D7EE",
    "grid": "#DDE6F6",
    "pill_bg": "#E8F2FF",
    "pill_text": "#2F6EA9",
    "card_title": "#0F3D66",
    "good_bg": "#E6F6ED",
    "good_border": "#CDEEDD",
    "good_text": "#174D2E",
    "mod_bg": "#FFF6E0",
    "mod_border": "#FFE7B8",
    "mod_text": "#6A4B00",
    "bad_bg": "#FDECEA",
    "bad_border": "#F5C6CB",
    "bad_text": "#611A15",
    "stopped_bg": "#F3F6FB",
    "stopped_border": "#D5E3F4",
    "stopped_text": "#5A6B84",
    "monitor_bg": "#DDEBFF",
    "monitor_border": "#BBD2FF",
    "voice_neutral_bg": "#FFF7E6",
    "voice_neutral_border": "#FFE4B8",
    "voice_neutral_text": "#5A4A22",
    "voice_on_bg": "#E6F6ED",
    "voice_on_border": "#BEE8CF",
    "voice_on_text": "#174D2E",
    "accent_success": "#17A673",
    "accent_success_hover": "#159565",
    "recs_grad_top": "#F6FAFF",
    "recs_grad_bottom": "#EAF2FF",
    "dot_off": "#C6D7F2",
}

APP_FONT_FAMILY = 'Segoe UI'
APP_FONT_SIZE = 10

def apply_palette(widget):
    pal = widget.palette()
    pal.setColor(QPalette.Window, QColor(COLORS["bg_base"]))
    pal.setColor(QPalette.Base, QColor(COLORS["bg_white"]))
    pal.setColor(QPalette.AlternateBase, QColor(COLORS["bg_alt"]))
    pal.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    pal.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    widget.setPalette(pal)
    widget.setAutoFillBackground(True)
    widget.setFont(QFont(APP_FONT_FAMILY, APP_FONT_SIZE))

APP_QSS = f"""
* {{ font-family: "{APP_FONT_FAMILY}", "Helvetica Neue", Arial; }}
QMainWindow {{ background-color: {COLORS['bg_base']}; }}
QLabel {{ color: {COLORS['text_primary']}; }}
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    background-color: {COLORS['bg_white']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 2px 4px;
    color: {COLORS['brand']};
    font-weight: 600;
}}
QPushButton {{
    background: {COLORS['brand']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {COLORS['brand_hover']}; }}
QPushButton:disabled {{ background: #A9C3E6; color: #F0F6FF; }}
QSlider::groove:horizontal {{ height: 6px; background: {COLORS['border']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {COLORS['brand']}; width: 16px; height: 16px;
    margin: -6px 0; border-radius: 8px;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    background: {COLORS['bg_white']};
}}
QTabBar::tab {{
    background: #EAF2FF;
    color: {COLORS['text_primary']};
    padding: 8px 16px;
    margin: 6px 8px;
    border-radius: 8px;
    min-width: 160px;
    font-weight: 600;
}}
QTabBar::tab:selected {{ background: #D4E6FF; }}
QTabBar::tab:hover {{ background: #DDEBFF; }}
QTableWidget {{
    background: {COLORS['bg_white']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    gridline-color: {COLORS['grid']};
}}
QHeaderView::section {{
    background: {COLORS['bg_alt']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    font-weight: 600;
    padding: 6px;
}}
QScrollArea {{ border: none; background: transparent; }}
QLineEdit {{
    background: {COLORS['bg_white']};
    border: 1px solid {COLORS['border_soft']};
    border-radius: 6px;
    padding: 6px;
}}
QDoubleSpinBox, QSpinBox {{
    background: {COLORS['bg_white']};
    border: 1px solid {COLORS['border_soft']};
    border-radius: 6px;
    padding: 4px;
}}
QCheckBox {{ spacing: 6px; }}
"""

RECS_QSS = f"""
QWidget#RecsRoot {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {COLORS['recs_grad_top']}, stop:1 {COLORS['recs_grad_bottom']});
}}
QLabel.SectionTitle {{ color:{COLORS['brand']}; font-size:18px; font-weight:600; }}
QLabel.SubTitle {{ color:{COLORS['pill_text']}; font-size:14px; font-weight:600; }}
QPushButton.Primary {{
    background:{COLORS['brand']}; color:white; border:none; border-radius:8px; padding:8px 14px;
}}
QPushButton.Primary:hover {{ background:{COLORS['brand_hover']}; }}
QLineEdit {{
    background:{COLORS['bg_white']}; border:1px solid {COLORS['border_soft']}; border-radius:6px; padding:6px;
}}
QTableWidget {{
    background:{COLORS['bg_white']}; color:{COLORS['text_primary']}; gridline-color:{COLORS['grid']};
    border:1px solid {COLORS['border']}; border-radius:8px;
}}
QHeaderView::section {{
    background-color:{COLORS['bg_alt']}; color:{COLORS['text_primary']}; font-weight:600; border:1px solid {COLORS['border']};
}}
"""

PRODUCT_CARD_QSS = f"""
QFrame#ProductCard {{
    background: {COLORS['bg_white']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}
QLabel#CardTitle {{ color: {COLORS['card_title']}; font-weight: 600; font-size: 14px; }}
QLabel#Pill {{ color: {COLORS['pill_text']}; background: {COLORS['pill_bg']}; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
QLabel#Why {{ color: #3E4C5E; }}
QLabel#Meta {{ color: #5E6A7D; }}
QLabel#Price {{ color: {COLORS['brand']}; font-weight: 600; }}
"""

LIVE_IMAGE_QSS = f"""
QLabel {{
    background-color: {COLORS['bg_white']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    color: #5A6B84;
    padding: 8px;
}}
"""

STATUS_PANEL_QSS = f"""
QLabel {{
    background-color: {COLORS['bg_white']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 10px;
    color: {COLORS['text_muted']};
}}
"""

BTN_SUCCESS_QSS = f"QPushButton {{ background: {COLORS['accent_success']}; }} QPushButton:hover {{ background: {COLORS['accent_success_hover']}; }}"

POPUP_FRAME_QSS = f"QFrame {{ background-color: {COLORS['bg_white']}; border: 1px solid {COLORS['border']}; border-radius: 12px; }}"

DOT_QSS = f"""
QRadioButton::indicator {{ width: 10px; height: 10px; border-radius: 5px; background-color: {COLORS['dot_off']}; }}
QRadioButton::indicator:checked {{ background-color: {COLORS['brand']}; }}
"""

def posture_style(kind: str) -> str:
    if kind == "good":
        return f"""QLabel {{ background-color: {COLORS['good_bg']}; border: 1px solid {COLORS['good_border']}; border-radius: 10px; padding: 12px; color: {COLORS['good_text']}; font-weight: 600; }}"""
    if kind == "moderate":
        return f"""QLabel {{ background-color: {COLORS['mod_bg']}; border: 1px solid {COLORS['mod_border']}; border-radius: 10px; padding: 12px; color: {COLORS['mod_text']}; font-weight: 600; }}"""
    if kind == "bad":
        return f"""QLabel {{ background-color: {COLORS['bad_bg']}; border: 1px solid {COLORS['bad_border']}; border-radius: 10px; padding: 12px; color: {COLORS['bad_text']}; font-weight: 600; }}"""
    if kind == "monitor":
        return f"""QLabel {{ background-color: {COLORS['monitor_bg']}; border: 1px solid {COLORS['monitor_border']}; border-radius: 10px; padding: 12px; color: {COLORS['text_primary']}; font-weight: 600; }}"""
    
    # stopped/none
    return f"""QLabel {{ background-color: {COLORS['stopped_bg']}; border: 1px solid {COLORS['stopped_border']}; border-radius: 10px; padding: 12px; color: {COLORS['stopped_text']}; font-weight: 600; }}"""

def voice_status_style(mode: str) -> str:
    if mode == "on":
        return f"""QLabel {{ background-color: {COLORS['voice_on_bg']}; border: 1px solid {COLORS['voice_on_border']}; border-radius: 8px; padding: 8px; color: {COLORS['voice_on_text']}; }}"""
    if mode == "off":
        return f"""QLabel {{ background-color: {COLORS['bad_bg']}; border: 1px solid {COLORS['bad_border']}; border-radius: 8px; padding: 8px; color: {COLORS['bad_text']}; }}"""
    return f"""QLabel {{ background-color: {COLORS['voice_neutral_bg']}; border: 1px solid {COLORS['voice_neutral_border']}; border-radius: 8px; padding: 8px; color: {COLORS['voice_neutral_text']}; }}"""
