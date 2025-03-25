import json
import sys

import easygui
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QRadioButton, QTextEdit
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from file_handler import process_track, resolve_dir


class LightCurveAnnotator(QWidget):
    def __init__(self, tracks_file, filter_file, annotation_file, obj_number):
        super().__init__()

        self.annotation_file = annotation_file
        self.annotation_data = None
        self.load_annotation()

        self.obj_number = obj_number
        self.light_curves = process_track(tracks_file, filter_file)
        self.current_index = 0

        # Canvas with lc
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Info Display (QTextEdit)
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)
        self.info_display.setFont(QFont('Courier', 10))

        # Buttons
        self.prev_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')
        self.save_button = QPushButton('Save Changes')
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.save_button.clicked.connect(self.save_annotation)

        # Group box
        self.group_box = QGroupBox('')
        self.group_box_layout = QVBoxLayout()
        self.radio1 = QRadioButton('Not annotated')
        self.radio2 = QRadioButton('Glint')
        self.radio3 = QRadioButton('No glint')
        self.radio1.clicked.connect(self.update_annotation)
        self.radio2.clicked.connect(self.update_annotation)
        self.radio3.clicked.connect(self.update_annotation)

        self.group_box_layout.addWidget(self.radio1)
        self.group_box_layout.addWidget(self.radio2)
        self.group_box_layout.addWidget(self.radio3)
        self.group_box.setLayout(self.group_box_layout)

        # Layout
        layout1 = QVBoxLayout()
        layout1.addWidget(self.info_display)
        layout1.addWidget(self.group_box)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.canvas)
        layout2.addLayout(layout1)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.prev_button)
        layout3.addWidget(self.next_button)
        layout3.addWidget(self.save_button)

        layout4 = QVBoxLayout()
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)

        self.setLayout(layout4)
        self.setWindowTitle('LC annotation')
        self.show()
        self.update_image()

    def update_image(self):
        """Update QLabel with the current image."""
        if self.light_curves:
            print(f'{self.current_index+1}/{len(self.light_curves)}')
            self.set_checkbox()
            self.figure.clf()

            ax = self.figure.add_subplot(111)
            ax.invert_yaxis()
            period = self.light_curves[self.current_index].get_period()
            x = [x.mjd for x in self.light_curves[self.current_index].time_array]
            x = np.array([((i - x[0]) % period) / period for i in x])
            x[-1] = 1.0
            y = self.light_curves[self.current_index].mag_array

            ax.scatter(x, y)
            ax.set_title(f'track: {self.light_curves[self.current_index].track}, starting point: {self.light_curves[self.current_index].starting_point}')
            ax.set_xlabel('phase')
            ax.set_ylabel('magnitude')

            self.canvas.draw()

            self.info_display.clear()
            self.info_display.append(f'{self.current_index+1}/{len(self.light_curves)}')
            self.info_display.append(self.light_curves[self.current_index].print_lc())

        else:
            # self.image_label.setText("No images found")
            pass

    def show_previous(self):
        """Show the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def show_next(self):
        """Show the next image."""
        if self.current_index < len(self.light_curves) - 1:
            self.current_index += 1
            self.update_image()

    def load_annotation(self):
        with open(self.annotation_file) as file:
            self.annotation_data = json.load(file)

    def save_annotation(self):
        temp = {}
        for obj in self.annotation_data:
            for track in self.annotation_data[obj]:
                for start_point in self.annotation_data[obj][track]:
                    if self.annotation_data[obj][track][start_point]['glint'] is None:
                        continue

                    if obj not in temp:
                        temp[obj] = {}
                    if track not in temp[obj]:
                        temp[obj][track] = {}
                    if start_point not in temp[obj][track]:
                        temp[obj][track][start_point] = {}

                    temp[obj][track][start_point]['glint'] = self.annotation_data[obj][track][start_point]['glint']

        self.annotation_data = temp
        with open(self.annotation_file, 'w') as file:
            json.dump(self.annotation_data, file, indent=4)

    def update_annotation(self):
        obj = self.obj_number
        track = self.light_curves[self.current_index].track
        start_point = self.light_curves[self.current_index].starting_point

        if obj not in self.annotation_data:
            self.annotation_data[obj] = {}

        if track not in self.annotation_data[obj]:
            self.annotation_data[obj][track] = {}

        if start_point not in self.annotation_data[obj][track]:
            self.annotation_data[obj][track][start_point] = {}

        if self.radio1.isChecked():
            self.annotation_data[obj][track][start_point]['glint'] = None
        elif self.radio2.isChecked():
            self.annotation_data[obj][track][start_point]['glint'] = True
        elif self.radio3.isChecked():
            self.annotation_data[obj][track][start_point]['glint'] = False

    def set_checkbox(self):
        obj = self.obj_number
        track = self.light_curves[self.current_index].track
        start_point = self.light_curves[self.current_index].starting_point

        if obj not in self.annotation_data:
            self.radio1.setChecked(True)
            return

        if track not in self.annotation_data[obj]:
            self.radio1.setChecked(True)
            return

        if start_point not in self.annotation_data[obj][track]:
            self.radio1.setChecked(True)
            return

        if self.annotation_data[obj][track][start_point]['glint'] is True:
            self.radio2.setChecked(True)
        elif self.annotation_data[obj][track][start_point]['glint'] is False:
            self.radio3.setChecked(True)
        else:
            self.radio1.setChecked(True)


def ask_for_inout_dir():
    input_file = easygui.diropenbox('ab', 'cd', default=r'C:\Users\13and\PycharmProjects\DP\data\mmt')
    if input_file is None:
        print('Exiting')
        exit(0)

    return input_file


if __name__ == '__main__':
    input_dir = ask_for_inout_dir()
    p1, p2, num = resolve_dir(input_dir)
    p3 = r'C:\Users\13and\PycharmProjects\DP\data\annotation.json'
    app = QApplication(sys.argv)
    gallery = LightCurveAnnotator(p1, p2, p3, num)
    sys.exit(app.exec())
