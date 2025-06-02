import os.path
import os.path
import sys

import numpy as np
from PySide6.QtGui import QFont, Qt, QIntValidator
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QSlider,
                               QLineEdit, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.data_preprocessing.lc_generator import LightCurveGenerator
from src.utils.constants import data_directory


class LightCurveGeneratorGui(QWidget):
    def __init__(self):
        super().__init__()
        self.LCG = LightCurveGenerator()

        # Canvas with lc
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Info Display (QTextEdit)
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)
        self.info_display.setFont(QFont('Courier', 10))

        # Buttons
        self.gene_button = QPushButton('Generate Light Curve')
        self.save_button = QPushButton('Save Light Curve')
        self.glint_button = QPushButton('Add Glint')
        self.datas_button = QPushButton('Generate Dataset')
        self.gene_button.clicked.connect(self.generate_light_curve)
        self.save_button.clicked.connect(self.save_lc)
        self.glint_button.clicked.connect(self.add_glint_permanently)
        self.datas_button.clicked.connect(self.generate_dataset)

        # Line Edit
        self.line_edit = QLineEdit()
        self.int_validator = QIntValidator()
        self.line_edit.setValidator(self.int_validator)
        self.line_edit.setFont(QFont('Courier', 10))
        line_edit_label = QLabel('Number of Light Curves:')

        # Sliders
        self.slider0 = QSlider(Qt.Orientation.Horizontal)
        self.slider0.setRange(0, 100)
        self.slider0.setValue(50)
        self.slider0.valueChanged.connect(self.value_changed2)
        self.slider0_label = QLabel('Light Curves with Glint: 50%')

        self.slider1 = QSlider(Qt.Orientation.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(50)
        self.slider1.valueChanged.connect(self.value_changed)
        self.slider1_label = QLabel('Center: 0.5')

        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setRange(1, 100)
        self.slider2.setValue(30)
        self.slider2.valueChanged.connect(self.value_changed)
        self.slider2_label = QLabel('Sigma: 0.3')

        self.slider3 = QSlider(Qt.Orientation.Horizontal)
        self.slider3.setRange(0, 100)
        self.slider3.setValue(5)
        self.slider3.valueChanged.connect(self.value_changed)
        self.slider3_label = QLabel('Width: 0.05')

        # Layout
        layout1 = QVBoxLayout()
        layout1.addWidget(line_edit_label)
        layout1.addWidget(self.line_edit)
        layout1.addWidget(self.slider0_label)
        layout1.addWidget(self.slider0)
        layout1.addWidget(self.datas_button)
        layout1.addWidget(self.info_display)
        layout1.addWidget(self.slider1_label)
        layout1.addWidget(self.slider1)
        layout1.addWidget(self.slider2_label)
        layout1.addWidget(self.slider2)
        layout1.addWidget(self.slider3_label)
        layout1.addWidget(self.slider3)
        layout1.addWidget(self.glint_button)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.canvas)
        layout2.addLayout(layout1)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.gene_button)
        layout3.addWidget(self.save_button)

        layout4 = QVBoxLayout()
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)

        self.setLayout(layout4)
        self.setWindowTitle('LC generator')
        self.show()

    def generate_light_curve(self):
        self.LCG.generate_light_curve(300, 0)
        self.show_light_curve()

    def generate_dataset(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select a directory', data_directory)
        if not dir_name:
            self.show_alert('fail', 'please select output directory')
            return

        if len(self.line_edit.text()) == 0:
            self.show_alert('fail', 'please select number of light curves')
            return

        count = int(self.line_edit.text())
        for i in range(count):
            glints = np.random.choice([0, 1, 2, 3], p=[.50, .35, .10, .05])
            self.LCG.generate_light_curve(300, glints)
            self.LCG.save_light_curve_img(dir_name, i)
            self.LCG.save_light_curve_txt(dir_name, i)

        self.show_alert('success', f'dataset saved to: {dir_name}')

    def save_lc(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select a directory', data_directory)
        if not dir_name:
            return

        name = len(os.listdir(dir_name)) + 1
        self.LCG.save_light_curve_img(dir_name, name)
        self.LCG.save_light_curve_txt(dir_name, name)
        self.show_alert('success', f'light curve saved to: {dir_name}')

    def add_glint_test(self):
        center = self.slider1.value() / 100
        sigma = self.slider2.value() / 100
        width = self.slider3.value() / 100
        self.LCG.add_delta_glint(center, sigma, width)
        self.show_light_curve()

    def add_glint_permanently(self):
        self.LCG.merge_specular()
        self.show_light_curve()

    def show_light_curve(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.invert_yaxis()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Magnitude [mag]')
        ax.scatter(self.LCG.x, self.LCG.y_diffuse + self.LCG.y_noise + self.LCG.y_specular, label='Diffuse part')
        ax.plot(self.LCG.x[self.LCG.y_specular_test < -0.01], (self.LCG.y_specular_test + self.LCG.y_diffuse)[self.LCG.y_specular_test < -0.01], '-or', label='Specular part')

        # ax.scatter(self.LCG.x, np.zeros(self.LCG.x.shape), label='Diffuse part')
        # ax.plot(self.LCG.x[self.LCG.y_specular_test < -0.01],
        #         self.LCG.y_specular_test[self.LCG.y_specular_test < -0.01], '-or', label='Specular part')

        # temp = [x/100 for x in range(13, 40)]
        # center = 0.5
        # width = 0.05
        # temp2 = [np.max(np.exp(-((self.LCG.x - center) / (width/4)) ** 2) / (x * np.sqrt(np.pi))) for x in temp]
        # ax.plot(temp, temp2)

        ax.legend()

        self.canvas.draw()

    def value_changed(self):
        self.slider1_label.setText(f'{self.slider1_label.text().split(":")[0]}: {self.slider1.value()/100}')
        self.slider2_label.setText(f'{self.slider2_label.text().split(":")[0]}: {self.slider2.value()/100}')
        self.slider3_label.setText(f'{self.slider3_label.text().split(":")[0]}: {self.slider3.value()/100}')
        self.LCG.clear_glints()
        self.add_glint_test()

    def value_changed2(self):
        self.slider0_label.setText(f'{self.slider0_label.text().split(":")[0]}: {self.slider0.value()}%')

    def show_alert(self, title, message):
        alert = QMessageBox(self)
        alert.setWindowTitle(title)
        alert.setText(message)
        alert.setIcon(QMessageBox.Information)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gallery = LightCurveGeneratorGui()
    sys.exit(app.exec())
