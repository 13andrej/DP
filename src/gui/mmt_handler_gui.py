import sys

from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel,
                               QLineEdit, QFileDialog, QMessageBox, QGroupBox, QRadioButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.data_preprocessing.mmt_handler import MMTHandler
from src.utils.constants import data_directory


class MMTHandlerGui(QWidget):
    def __init__(self):
        super().__init__()
        self.MMTH = MMTHandler()
        self.current_index = 0

        # Canvas with lc
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Info Display
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)
        self.info_display.setFont(QFont('Courier', 10))

        # Line Edit
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText('300')
        self.int_validator = QIntValidator()
        self.line_edit.setValidator(self.int_validator)
        self.line_edit.setFont(QFont('Courier', 10))
        line_edit_label = QLabel('Number of points:')

        # Group box
        self.group_box1 = QGroupBox('magnitude')
        self.radio1_1 = QRadioButton('normal')
        self.radio1_2 = QRadioButton('reduced')
        self.radio1_1.toggled.connect(self.update_scene)
        self.radio1_2.toggled.connect(self.update_scene)
        self.radio1_1.setChecked(True)
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.radio1_1)
        vbox1.addWidget(self.radio1_2)
        vbox1.addStretch(1)
        self.group_box1.setLayout(vbox1)

        self.group_box2 = QGroupBox('interpolate')
        self.radio2_1 = QRadioButton('no')
        self.radio2_2 = QRadioButton('yes')
        self.radio2_1.toggled.connect(self.update_scene)
        self.radio2_2.toggled.connect(self.update_scene)
        self.radio2_1.setChecked(True)
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.radio2_1)
        vbox2.addWidget(self.radio2_2)
        vbox2.addWidget(line_edit_label)
        vbox2.addWidget(self.line_edit)
        vbox2.addStretch(1)
        self.group_box2.setLayout(vbox2)

        # Buttons
        self.prev_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')
        self.save_button = QPushButton('Save Light Curve')
        self.save_all_button = QPushButton('Save All')
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.save_button.clicked.connect(self.save_lc)
        self.save_all_button.clicked.connect(self.save_all)

        # Layout
        layout1 = QVBoxLayout()
        layout1.addWidget(self.info_display)
        layout1.addWidget(self.save_button)
        layout1.addWidget(self.save_all_button)
        layout1.addWidget(self.group_box1)
        layout1.addWidget(self.group_box2)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.canvas)
        layout2.addLayout(layout1)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.prev_button)
        layout3.addWidget(self.next_button)

        layout4 = QVBoxLayout()
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)

        self.setLayout(layout4)
        self.setWindowTitle('MMT dataset handler')
        self.show()
        self.load_light_curves()

    def load_light_curves(self):
        annotation_file, _ = QFileDialog.getOpenFileName(self, 'Select an annotation file', data_directory)
        if not annotation_file:
            print('Exiting')
            exit(0)

        self.MMTH.load_annotation_meta(annotation_file)
        self.current_index = 0
        self.update_scene()

    def update_scene(self):
        if len(self.MMTH.light_curves_meta) == 0:
            return

        self.MMTH.load_specific_lc1(self.current_index)
        self.figure.clf()

        if self.radio1_1.isChecked():
            self.MMTH.get(self.current_index).compute_normal()

        elif self.radio1_2.isChecked():
            self.MMTH.get(self.current_index).compute_reduced()

        if self.radio2_2.isChecked():
            n = int(self.line_edit.text()) if len(self.line_edit.text()) else int(self.line_edit.placeholderText())
            self.MMTH.get(self.current_index).compute_interpolated(n)

        ax = self.figure.add_subplot(111)
        self.MMTH.get(self.current_index).plot_gui(ax)
        self.set_gui_features()
        self.canvas.draw()

    def save_lc(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select a directory', data_directory)
        if not dir_name:
            return

        self.MMTH.get(self.current_index).save_img(dir_name)
        self.MMTH.get(self.current_index).save_txt(dir_name)
        self.show_alert('success', f'light curve saved to: {dir_name}')

    def save_all(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select a directory', data_directory)
        if not dir_name:
            return

        for i in range(self.MMTH.size()):
            if self.MMTH.get(i) is None:
                continue
            self.MMTH.get(i).save_img(dir_name)
            self.MMTH.get(i).save_txt(dir_name)
        self.show_alert('success', f'light curves saved to: {dir_name}')

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_scene()

    def show_next(self):
        if self.current_index < self.MMTH.size() - 1:
            self.current_index += 1
            self.update_scene()

    def set_gui_features(self):
        self.info_display.clear()
        self.info_display.append(f'{self.current_index + 1}/{self.MMTH.size()}')
        self.info_display.append(self.MMTH.get(self.current_index).print())

    def show_alert(self, title, message):
        alert = QMessageBox(self)
        alert.setWindowTitle(title)
        alert.setText(message)
        alert.setIcon(QMessageBox.Information)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gallery = MMTHandlerGui()
    sys.exit(app.exec())
