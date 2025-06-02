import json
import os.path
import sys

import numpy as np
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QRadioButton,
                               QTextEdit, QComboBox, QLabel, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.utils.constants import data_directory, mmt_directory
from src.utils.file_handler import read_tracks_file, get_header


class LightCurveAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.annotation_file = ''
        self.obj_number = ''
        self.header = []
        self.light_curves = []
        self.current_index = 0
        self.annotation_data = {}

        # Canvas with lc
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Info Display
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)
        self.info_display.setFont(QFont('Courier', 10))

        # Comment
        self.comment = QTextEdit(self)
        self.comment.setFont(QFont('Courier', 10))
        comment_label = QLabel('Comment: ')

        # Buttons
        self.prev_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')
        self.skip_button = QPushButton('Skip Track')
        self.save_button = QPushButton('Save Changes')
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        self.skip_button.clicked.connect(self.skip_track)
        self.save_button.clicked.connect(self.save_annotation)

        # Group box
        self.group_box = QGroupBox('')
        self.radio1 = QRadioButton('Not annotated')
        self.radio2 = QRadioButton('Glint')
        self.radio3 = QRadioButton('No glint')
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(self.radio1)
        group_box_layout.addWidget(self.radio2)
        group_box_layout.addWidget(self.radio3)
        self.group_box.setLayout(group_box_layout)

        # Combo box
        self.combo_box = QComboBox()
        self.combo_box.addItem('1', 1)
        self.combo_box.addItem('2', 2)
        self.combo_box.addItem('3', 3)
        self.combo_box.addItem('4', 4)
        combo_label = QLabel('Number of glints: ')

        # Layout
        layout1 = QVBoxLayout()
        layout1.addWidget(self.info_display)
        layout1.addWidget(comment_label)
        layout1.addWidget(self.comment)
        layout1.addWidget(combo_label)
        layout1.addWidget(self.combo_box)
        layout1.addWidget(self.group_box)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.canvas)
        layout2.addLayout(layout1)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.prev_button)
        layout3.addWidget(self.next_button)
        layout3.addWidget(self.skip_button)
        layout3.addWidget(self.save_button)

        layout4 = QVBoxLayout()
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)

        self.setLayout(layout4)
        self.setWindowTitle('LC annotation')
        self.show()
        self.load_light_curves()

    def load_light_curves(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select a directory', mmt_directory)
        if not dir_name:
            print('Exiting')
            exit(0)

        annotation_file, _ = QFileDialog.getOpenFileName(self, 'Select an annotation file', data_directory)
        if not annotation_file:
            print('Exiting')
            exit(0)

        num = os.path.basename(dir_name).split('_')[-1]
        tracks_file = os.path.join(dir_name, f'{num}_tracks.txt')
        filter_file = os.path.join(dir_name, 'Filter')

        self.annotation_file = annotation_file
        self.obj_number = num
        self.header = get_header(tracks_file)
        self.light_curves = read_tracks_file(tracks_file, filter_file)
        self.current_index = 0

        with open(self.annotation_file) as file:
            self.annotation_data = json.load(file)

        self.update_scene()

    def show_alert(self, title, message):
        alert = QMessageBox(self)
        alert.setWindowTitle(title)
        alert.setText(message)
        alert.setIcon(QMessageBox.Information)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec()

    def update_scene(self):
        """Draw current light curve"""
        if len(self.light_curves) == 0:
            return

        self.set_gui_features()
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
        ax.set_ylabel('Magnitude [mag]')
        ax.set_xlabel('Phase')

        self.canvas.draw()

    def show_previous(self):
        """Show the previous light curve"""
        if self.current_index > 0:
            self.update_annotation()
            self.current_index -= 1
            self.update_scene()

    def show_next(self):
        """Show the next light curve"""
        if self.current_index < len(self.light_curves) - 1:
            self.update_annotation()
            self.current_index += 1
            self.update_scene()

    def skip_track(self):
        """skip all light curves from current track"""
        ind = self.current_index

        while self.light_curves[ind].track == self.light_curves[self.current_index].track:
            ind += 1
            if ind >= len(self.light_curves):
                return

        self.update_annotation()
        self.current_index = ind
        self.update_scene()

    def save_annotation(self):
        """save annotation into file"""
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
                    temp[obj][track][start_point]['P[s]'] = self.annotation_data[obj][track][start_point]['P[s]']

                    if 'comment' in self.annotation_data[obj][track][start_point]:
                        temp[obj][track][start_point]['comment'] = self.annotation_data[obj][track][start_point]['comment']

        self.annotation_data = temp
        with open(self.annotation_file, 'w') as file:
            json.dump(self.annotation_data, file, indent=4)

        self.show_alert('success', f'annotation saved to: {self.annotation_file}')

    def update_annotation(self):
        """add changes into dict object"""
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
            self.annotation_data[obj][track][start_point]['glint'] = self.combo_box.currentData()
        elif self.radio3.isChecked():
            self.annotation_data[obj][track][start_point]['glint'] = 0

        if self.comment.toPlainText() != '':
            self.annotation_data[obj][track][start_point]['comment'] = self.comment.toPlainText()

        self.annotation_data[obj][track][start_point]['P[s]'] = self.light_curves[self.current_index].get_period_seconds()

    def set_gui_features(self):
        """"""
        obj = self.obj_number
        track = self.light_curves[self.current_index].track
        start_point = self.light_curves[self.current_index].starting_point

        self.info_display.clear()
        self.info_display.append(f'{self.current_index + 1}/{len(self.light_curves)}')
        self.info_display.append(self.light_curves[self.current_index].print_lc())
        self.info_display.append('\n'.join(self.header[:5]))
        self.comment.setPlainText('')
        self.combo_box.setCurrentIndex(0)

        if obj not in self.annotation_data:
            self.radio1.setChecked(True)
            return

        if track not in self.annotation_data[obj]:
            self.radio1.setChecked(True)
            return

        if start_point not in self.annotation_data[obj][track]:
            self.radio1.setChecked(True)
            return

        if self.annotation_data[obj][track][start_point]['glint'] is None:
            self.radio1.setChecked(True)
        elif self.annotation_data[obj][track][start_point]['glint'] > 0:
            self.radio2.setChecked(True)
            self.combo_box.setCurrentIndex(self.annotation_data[obj][track][start_point]['glint']-1)
        else:
            self.radio3.setChecked(True)

        self.comment.setPlainText(self.annotation_data[obj][track][start_point].get('comment'))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gallery = LightCurveAnnotator()
    sys.exit(app.exec())
