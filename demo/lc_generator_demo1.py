from src.data_preprocessing.lc_generator import LightCurveGenerator


if __name__ == '__main__':
    LCG = LightCurveGenerator()
    LCG.generate_light_curve(300, 3)
    LCG.show_light_curve()
