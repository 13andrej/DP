import os

from src.utils.constants import data_directory, mmt_directory, no_of_points
from src.utils.file_handler import read_annotation_meta, read_filter_meta_dir, read_specific_lc


class MMTHandler:
    def __init__(self):
        self.light_curves_meta = []
        self.light_curves = {}

    def load_annotation_meta(self, annotation_file):
        """load metadata from light curves marked in annotation file"""
        self.light_curves_meta = read_annotation_meta(annotation_file)

    def load_all_meta(self):
        """load all metadata from mmt directory"""
        self.light_curves_meta = read_filter_meta_dir(mmt_directory)

    def load_specific_lc1(self, index):
        """load light curve defined with index in light curves metadata"""
        if self.light_curves_meta[index][:3] in self.light_curves:
            return
        o, t, s, g, p = self.light_curves_meta[index]
        self.light_curves[(o, t, s)] = read_specific_lc(o, t, s)
        self.light_curves[(o, t, s)].set_values(glints=g, period=p)

    def load_specific_lc2(self, obj_number, track_number, starting_point, glints=None, period=None):
        """load light curve defined with obj_number, track_number, starting_point"""
        if (obj_number, track_number, starting_point) in self.light_curves:
            return
        self.light_curves[(obj_number, track_number, starting_point)] = read_specific_lc(obj_number, track_number, starting_point)
        self.light_curves[(obj_number, track_number, starting_point)].set_values(glints=glints, period=period)

    def load_all_lc(self):
        """load all light curves which are in metadata"""
        for o, t, s, g, p in self.light_curves_meta:
            self.load_specific_lc2(o, t, s, g, p)

    def get(self, index):
        """return light curve defined with index in light curves array"""
        if self.light_curves_meta[index][:3] not in self.light_curves:
            return
        return self.light_curves[self.light_curves_meta[index][:3]]

    def get2(self, obj_number, track_number, starting_point):
        """return light curve defined with obj_number, track_number, starting_point"""
        if (obj_number, track_number, starting_point) not in self.light_curves:
            return
        return self.light_curves[(obj_number, track_number, starting_point)]

    def size(self):
        """return number of loaded light curves metadata"""
        return len(self.light_curves_meta)


def save_all():
    """process and save all light curves marked in annotation"""
    MMTH = MMTHandler()
    MMTH.load_annotation_meta(os.path.join(data_directory, 'annotation.json'))
    MMTH.load_all_lc()

    for lc in MMTH.light_curves.values():
        print(lc.get_name())
        lc.compute_interpolated(no_of_points)
        lc.save_img(os.path.join(data_directory, 'dataset', '04'))
        lc.save_txt(os.path.join(data_directory, 'dataset', '04'))


def save_not_saved():
    """process and save light curves marked in annotation which were not saved yet"""
    MMTH = MMTHandler()
    MMTH.load_annotation_meta(os.path.join(data_directory, 'annotation.json'))
    already_exists = 0

    for i in range(MMTH.size()):
        lc_name = f'{MMTH.light_curves_meta[i][0]}_{MMTH.light_curves_meta[i][1]}_{MMTH.light_curves_meta[i][2]}.txt'
        if os.path.exists(os.path.join(data_directory, 'dataset', '04', lc_name)):
            already_exists += 1
            continue

        MMTH.load_specific_lc1(i)
        lc = MMTH.get(i)
        print(lc.get_name())
        lc.compute_interpolated(no_of_points)
        lc.save_img(os.path.join(data_directory, 'dataset', '04'))
        lc.save_txt(os.path.join(data_directory, 'dataset', '04'))

    print(f'skipping {already_exists} light curves which already exists')


if __name__ == '__main__':
    save_not_saved()
