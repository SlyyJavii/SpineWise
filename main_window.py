from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.uic import loadUi
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the UI file here
        loadUi("my_ui_file.ui", self)
        
        self.setWindowTitle("SpineWise UI")
        self.setGeometry(100, 100, 600, 400)

        # Optional: extra widget setup if needed
        # label = QLabel("Welcome to SpineWise!", self)
        # label.move(220, 180)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
