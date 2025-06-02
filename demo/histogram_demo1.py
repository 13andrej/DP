from src.data_preprocessing.histogram_builder import HistogramBuilder


if __name__ == '__main__':
    HB = HistogramBuilder()
    HB.load_coefficients_results()
    # HB.make_no_points_histogram()
    # HB.make_rms_histogram()
    # HB.compare_2_elements('a0', 'a1')
    # HB.make_no_points_histogram()
    HB.compare_all_elements()
