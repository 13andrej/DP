import os

import numpy as np
import oct2py


exports_path = r'C:\Users\13and\PycharmProjects\DP\data\dataset\02\exports'
output_dir = r'C:\Users\13and\PycharmProjects\DP\data\peakfinder'


def find_peaks(slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type):
    oc = oct2py.Oct2Py()
    oc.addpath(r'./Octave')  # 'C:\Users\13and\Developer\Octave'

    res = []

    for file in os.listdir(exports_path):
        file_path = os.path.join(exports_path, file)
        a, b = oc.runScript3(file_path, slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type, nout=2)
        res.append((file, a, b))

    oc.close()

    res = [(a, float(b[0][1]) if b[0][0] != 0 else 'None', float(c) if c != 'None' else c) for a, b, c in res]
    output_path = os.path.join(output_dir, f'slope={slope_threshold}amp={amp_threshold}smooth_width={smooth_width}.txt')
    save_results(output_path, res)
    print(evaluate2(res))


def save_results(file_path, result):
    with open(file_path, 'w') as file:
        for entry in result:
            file_name, temp1, temp2 = entry
            # temp1 = temp1[0][1] if temp1[0][0] != 0 else 'None'
            file.write(f'{file_name}\t{temp1}\t{temp2}\n')


def evaluate(result):
    good_found = 0
    good_not_found = 0
    bad_found = 0
    bad_not_found = 0

    diff = []

    for (file_name, found, truth) in result:
        if truth == 'None':
            if found == 'None':
                good_not_found += 1
            else:
                bad_found += 1
        else:
            if found == 'None':
                bad_not_found += 1
            else:
                good_found += 1
                diff.append(abs(float(found) - float(truth)))

    print(f'good: {good_found + good_not_found}\nbad: {bad_found + bad_not_found}')
    print(f'good found: {good_found}\ngood not found: {good_not_found}')
    print(f'bad found: {bad_found}\nbad not found: {bad_not_found}')
    print(sum(diff) / (good_found + good_not_found))
    print(max(diff))


def evaluate2(result):
    tp, fp, fn = 0, 0, 0
    position_errors = []

    for file_name, detected, truth in result:
        if detected != 'None':
            if truth != 'None':
                tp += 1
                position_errors.append(abs(detected - truth))
            else:
                fp += 1
        elif truth != 'None':
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    mae = np.mean(position_errors) if position_errors else None

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "position_error": mae,
    }


def evaluate_file(file_path):
    res = []

    with open(file_path) as file:
        for line in file:
            file_name, found, truth = line.split()
            found = float(found) if found != 'None' else found
            truth = float(truth) if truth != 'None' else truth
            res.append((file_name, found, truth))

    return evaluate2(res)


def find_best_model():
    rec, res = {'precision': 0, 'recall': 0, 'f1_score': 0, 'position_error': 0}, None
    for x in os.listdir(output_dir):
        # print(x, end='   ')
        temp = evaluate_file(os.path.join(output_dir, x))
        if temp['f1_score'] > rec['f1_score']:
            rec = temp
            res = x

    print(res)
    print(rec)


if __name__ == '__main__':
    find_peaks(0.056, 10, 4, 10, 3)
    # evaluate_file(r'../data/peakfinder/slope=0.05amp=20smooth_width=5.txt')
    # find_best_model()

    # for slope in np.arange(0.06, 0.08, 0.002):
    #    for smooth_w in np.arange(4.0, 7.0, 1.0):
    #        print(f'slope: {slope} smooth width: {smooth_w}', end='    ')
    #        find_peaks(slope, 20, smooth_w, 10, 3)


