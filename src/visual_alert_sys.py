from PyQt5.QtWidgets import QApplication, QMessageBox


def show_notification(title: str, message: str) -> None:
    """Display a warning popup with the given title and message."""
    app = QApplication.instance()
    close_app = False
    if app is None:
        # If no QApplication has been created, create one.
        app = QApplication([])
        close_app = True

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec_()

    if close_app:
        app.quit()
