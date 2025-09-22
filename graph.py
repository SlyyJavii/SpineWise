import sys
import csv
import matplotlib
from threading import *
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

pos = 0
oldPos = pos

class Worker(QObject):
    # constructor provides function and arguments
    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    # thread executes provided functions(arguments)
    def run(self):
        self.fn(*self.args)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)
        self.file = None
        # basic file exception handling
        try:
            self.file = open("posture_trend_log.csv", "rb")
        except FileNotFoundError:
            raise
        # graph's x, y are initialized as 0
        self.xdata = [0] 
        self.ydata = [0]
        # without multithreading, program crashes because the GUI thread is always waiting for I/O
        # with multithreading, GUI thread is paused to give the other thread resources to execute
        self.thread = QThread()
        self.worker = Worker(self.update_plot)
        self.worker.moveToThread(self.thread)
        self.thread.start()

        # trigger the canvas to update and redraw based on timer
        self.timer = QtCore.QTimer()
        # delay of 1 sec
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.worker.run)
        self.timer.start()

    def update_plot(self):
            global pos, oldPos
            reader = csv.reader(polling_reader(self.file))
            for row in reader:
                # if row doesn't have 7 entries, switches main control back
                # confidence score is stored in 7th column
                # does not include empty entries
                if (len(row) != 6 or not (row[-1].isdigit()) ):
                    return
                (self.ydata).append(int(row[-1]))
                (self.xdata).append(self.xdata[-1]+3)
                # Clear the canvas.
                self.canvas.axes.cla()  
                self.canvas.axes.plot(self.xdata, self.ydata, 'r')
                # trigger the canvas to update and redraw
                self.canvas.draw()

# responsible for constantly polling for any updates to csv file
def polling_reader(file, encoding = "utf-8"):
    global pos
    # file is read in binary mode, so that:
    # file pointer can be modified and referenced
    while True:
        file.seek(pos)
        for line in file:
            if line.strip():
                yield line.decode("utf-8")
        # if old file pointer == new file pointer
        # no new row has been added, so keep polling
        if (pos != file.tell()):
            pos = file.tell()
            break

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()

    # to test this py script, launch the file, and continually make changes to the "posture_trend_log.csv", then save
    # whatever is saved, as long as it is a numerical value in the 7th row, should be reflected to the graph