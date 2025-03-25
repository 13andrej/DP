from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QTextEdit
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PySide6 GUI with Matplotlib and Info Panel")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib Figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Info Display (QTextEdit)
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)  # Make it read-only
        self.info_display.setFont(QFont("Courier", 10))
        layout.addWidget(self.info_display)

        # Button to update plot and info
        self.button = QPushButton("Generate Plot", self)
        self.button.clicked.connect(self.update_plot)
        layout.addWidget(self.button)

    def update_plot(self):
        """Generate a sine wave plot and display info."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label="Sine Wave", color="blue")

        ax.set_title("Sine Wave")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()

        self.canvas.draw()  # Update plot

        # Update info panel
        self.info_display.append("Plot updated with sine wave data.")
        self.info_display.append(f"X range: {x[0]} to {x[-1]}")
        self.info_display.append("Y values between: {:.2f} and {:.2f}".format(min(y), max(y)))
        self.info_display.append("-" * 40)  # Separator line


if __name__ == "__main__":
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec()
