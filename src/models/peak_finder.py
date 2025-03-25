import os

import numpy as np
import oct2py


def run_peak_finder(exports_path, slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type, output_directory=None, verbose=False):
    oc = oct2py.Oct2Py()
    oc.addpath(r'./Octave')

    res = []

    if os.path.isdir(exports_path):
        for file in os.listdir(exports_path):
            file_path = os.path.join(exports_path, file)
            a, b = oc.runScript3(file_path, slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type, nout=2)
            res.append((file, a, b))
            print(file, a, b) if verbose else None
    else:
        a, b = oc.runScript3(exports_path, slope_threshold, amp_threshold, smooth_width, peak_group, smooth_type, nout=2)
        res.append((exports_path, a, b))
        print(exports_path, a, b) if verbose else None

    oc.close()

    res = [(a, float(b[0][1]) if b[0][0] != 0 else 'None', float(c) if c != 'None' else c) for a, b, c in res]
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        save_path = os.path.join(output_directory, f'slope={round(slope_threshold, 3)}amp={amp_threshold}smooth_width={round(smooth_width, 1)}.txt')
        save_results(save_path, res)

    return res


def save_results(save_path, result):
    with open(save_path, 'w') as file:
        for entry in result:
            file_name, temp1, temp2 = entry
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

    return {
        'good': good_found + good_not_found,
        'good found': good_found,
        'good not found': good_not_found,
        'bad': bad_found + bad_not_found,
        'bad found': bad_found,
        'bad not found': bad_not_found,
        'mean diff': sum(diff) / (good_found + good_not_found),
        'max diff': max(diff)
    }


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
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'position_error': mae,
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


def perform_grid_search(exports_path, output_directory=None, verbose=False):
    res = {'f1_score': 0.0, 'slope': None, 'smooth width': None}

    for slope in np.arange(0.052, 0.06, 0.002):
        for smooth_w in np.arange(4.0, 5.0, 0.2):
            temp = run_peak_finder(exports_path, slope, 20, smooth_w, 10, 3, output_directory=output_directory)
            temp_eval = evaluate2(temp)

            if verbose:
                print(f'slope: {slope} smooth width: {smooth_w}    ', temp_eval)

            if temp_eval['f1_score'] > res['f1_score']:
                res['f1_score'] = temp_eval['f1_score']
                res['slope'] = slope
                res['smooth width'] = smooth_w

    return res


def find_best_model(models_directory, verbose=False):
    res = {'file': None, 'precision': 0, 'recall': 0, 'f1_score': 0, 'position_error': 0}

    for file in os.listdir(models_directory):
        temp = evaluate_file(os.path.join(models_directory, file))
        
        if verbose:
            print(file, temp)

        if temp['f1_score'] > res['f1_score']:
            res = temp
            res['file'] = file

    return res


if __name__ == '__main__':
    exp_dir = r'C:\Users\13and\PycharmProjects\DP2\data\dataset\01\exports'
    out_dir = r'C:\Users\13and\PycharmProjects\DP2\data\peak_finder'

    # print(perform_grid_search(exp_dir, out_dir, verbose=True))
    print(find_best_model(out_dir))
