import io
import tempfile

import easygui
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

from PIL import Image
from astropy.time import Time
import pandas as pd
import os


def process_track(tracks_file, filter_file):
    with open(filter_file) as file:
        skip_lines = 1
        starting_points = {}

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            _, track, starting_point = line.strip().split()

            if track not in starting_points:
                starting_points[track] = []
            starting_points[track].append(int(starting_point))

    with open(tracks_file) as file:
        skip_lines = 7
        last_track, counter, start_point, time_array, mag_array = None, 0, 0, [], []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            date, time, _, mag, _, _, d, _, _, track = line.strip().split()

            if last_track != track:
                if last_track is not None:
                    starting_points[last_track].append(9999999)
                    for i in range(len(starting_points[last_track]) - 1):
                        a, b = starting_points[last_track][i], starting_points[last_track][i+1]
                        plt.clf()
                        plt.gca().invert_yaxis()
                        plt.plot([x.mjd for x in time_array][a: b], mag_array[a: b])
                        # plt.show()

                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            plt.savefig(tmpfile.name)
                            # plt.close(fig)  # Free memory
                            # easygui.msgbox("Light Curve", image=tmpfile.name)
                            easygui.buttonbox(msg='aa', title='decision', choices=['yes', 'no'], default_choice='yes',
                                              cancel_choice='Exit', image=tmpfile.name)
                    exit(99)

                time_array, mag_array = [], []
                last_track = track
                counter = 0

            time_array.append(Time(f'{date}T{time}', format='isot'))
            mag_array.append(float(mag))
            counter += 1


def ask_for_inout_file():
    input_file = easygui.fileopenbox('ab', 'cd', default=r'C:\Users\13and\PycharmProjects\DP\data\mmt')
    if input_file is None:
        print('Exiting')
        exit(0)

    return input_file


if __name__ == '__main__':
    # Run the annotation tool
    # root = tk.Tk()
    # app = LightCurveAnnotator(root)
    # root.mainloop()

    # ask_for_inout_file()
    # easygui.buttonbox(msg='aa', title='decision', choices=['yes', 'no'], default_choice='yes', cancel_choice='Exit',
    #                   image=r'C:\Users\13and\Desktop\pictures\-\Nový priečinok\flhvwkms78na1.jpg')

    process_track(r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13\13_tracks.txt',
                  r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13\Filter')
