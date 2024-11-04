import os
import oct2py


if __name__ == '__main__':
    oc = oct2py.Oct2Py()
    oc.addpath(r'./Octave')  # 'C:\Users\13and\Developer\Octave'

    res = []

    dir_path = r'C:\Users\13and\PycharmProjects\DP\data\dataset\exports'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        a, b = oc.runScript2(file_path, nout=2)
        res.append((file, a, b))

    oc.close()

    with open(r'C:\Users\13and\PycharmProjects\DP\data\find_peaks.txt', 'w') as file:
        for entry in res:
            file_name, temp1, temp2 = entry
            temp1 = temp1[0][1] if temp1[0][0] != 0 else 'None'
            file.write(f'{file_name}\t{temp1}\t{temp2}\n')
