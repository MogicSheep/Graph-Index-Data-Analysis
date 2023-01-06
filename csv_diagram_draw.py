import numpy as np
import csv
import struct
from scipy.optimize import curve_fit
import matplotlib as mlib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from energy_calc import generate_new_line

if __name__ == "__main__":

    filename = 'D:\\Desktop\\Graph base method\\Energy_Base_Search\\cmake-build-debug\\test\\cluster_energy_result.csv'
    filename = 'D:\\Desktop\\Graph base method\\Search Algorithm\\cmake-build-debug\\test\\1000_cluster.csv'
    filename = 'D:\\Desktop\\Graph base method\\Data_Driven_Search\\cmake-build-debug\\test\\pruning.csv'
    data = {}
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            method_name = row[0]
            Qbs = row[1]
            recall = row[2]
            if method_name not in data.keys():
                data[method_name] = []
            data[method_name].append((Qbs, recall))

    color_tabel = ['r', 'b', 'g', 'y']
    shape_table = ['*', '^', 'd', 'o']
    count = 0
    for method_name, data_list in data.items():
        date_len = len(data_list)
        Qbs = np.zeros(date_len)
        recall = np.zeros(date_len)

        data_list.sort(key=lambda x: x[1])
        print(method_name)
        for j in range(0, date_len):
            Qbs[j] = float(data_list[j][0])
            recall[j] = float(data_list[j][1])
            print(recall[j])
            print(Qbs[j])

        col = color_tabel[count]
        shp = shape_table[count]
        count += 1
        plt.plot(recall, Qbs, col, marker = shp, label=method_name)

    plt.xlabel('recall@1')
    plt.ylabel('Qbs')
    plt.legend(loc=3)
    plt.title('gist1m_nsg_pruning256')
    plt.show()

#
# 1 987636
# 124761 889819 532108 987074 987215 20845 517238 833054 987636
#
