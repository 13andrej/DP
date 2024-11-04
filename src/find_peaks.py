import os
import oct2py


def find_peaks():
    oc = oct2py.Oct2Py()
    oc.addpath(r'./Octave')  # 'C:\Users\13and\Developer\Octave'

    res = []

    dir_path = r'C:\Users\13and\PycharmProjects\DP\data\dataset\exports'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        a, b = oc.runScript2(file_path, nout=2)
        res.append((file, a, b))

    oc.close()

    with open(r'C:\Users\13and\PycharmProjects\DP\data\peakfinder\find_peaks.txt', 'w') as file:
        for entry in res:
            file_name, temp1, temp2 = entry
            temp1 = temp1[0][1] if temp1[0][0] != 0 else 'None'
            file.write(f'{file_name}\t{temp1}\t{temp2}\n')


def evaluate():
    good = 0
    bad_found = 0
    bad_not_found = 0

    diff = []

    with open(r'C:\Users\13and\PycharmProjects\DP\data\find_peaks.txt') as file:
        for line in file:
            file_name, found, truth = line.split()

            if truth == 'None':
                if found == 'None':
                    good += 1
                else:
                    bad_found += 1
            else:
                if found == 'None':
                    bad_not_found += 1
                else:
                    good += 1
                    diff.append(abs(float(found) - float(truth)))

    print(f'good: {good}\nbad: {bad_found + bad_not_found}')
    print(f'bad found: {bad_found}\nbad not found: {bad_not_found}')
    print(sum(diff) / good)
    print(max(diff))


if __name__ == '__main__':
    # find_peaks()
    evaluate()
