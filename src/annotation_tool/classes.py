

class LC:
    def __init__(self, time_array, mag_array, track, starting_point):
        self.time_array = time_array
        self.mag_array = mag_array
        self.track = track
        self.starting_point = starting_point

    def print_lc(self):
        # print(self.track, self.starting_point)
        return f'number of points: {len(self.time_array)}'

    def get_period(self):
        return self.time_array[-1].mjd - self.time_array[0].mjd
