import numpy as np
import matplotlib as mlib
import matplotlib.pyplot as plt
import json
import csv


def load_test_data(file_name):
    data_ = {}
    with open(file_name, 'r') as f:
        for line in f:
            if float(line) < 1000:
                node_id = int(line)
                data_[node_id] = []
            else:
                data_[node_id].append(float(line))

    return data_


def load_bad_data(file_name):
    data_ = {}
    with open(file_name, 'r') as f:
        for line in f:
            if line.split(',')[0] == 'node':
                node_id = int(line.split(',')[1])
                data_[node_id] = []
            else: 
                data_[node_id].append(float(line))

    return data_


if __name__ == '__main__':
    print("begin data analysis")
    mlib.rcParams['legend.fontsize'] = 10
    search_data = load_bad_data("test_bad.log")
    #search_data = load_bad_data("test_momentum.log")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlabel("node_id")
    plt.ylabel("node_iteration")
    limited = 10
    for key, list in search_data.items():
        limited -= 1
        if limited == 0 :
            break
        cnt = 0
        size = len(list)
        x = np.zeros(size)
        y = np.zeros(size)
        z = np.zeros(size)
        for value in list:
            x[cnt] = int(key)
            y[cnt] = cnt
            z[cnt] = float(value)
            cnt += 1
        ax.plot(x, y, z)
    ax.legend()
    plt.show()
