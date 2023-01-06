import numpy as np
import csv
import struct
from scipy.optimize import curve_fit
import matplotlib as mlib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from energy_calc import generate_new_line


def curve_line(label_x_t, label_y_t, label_x_q, label_y_q, item_id):
    if len(label_x_t) > 0:
        # z1 = np.polyfit(label_x_t, label_y_t, 2)  # 用3次多项式拟合
        # p1 = np.poly1d(z1)
        # tag = generate_new_line(z1, label_x_t, label_y_t, 0.9999)
        # print(p1)  # 在屏幕上打印拟合多项式
        # yvals = p1(label_x_t)  # 也可以使用yvals=np.polyval(z1,x)
        plt.plot(label_x_t, label_y_t, 'o', label='original train values')

        # z1[2] += tag
        # p1 = np.poly1d(z1)
        # yvals = p1(label_x_t)
        # plottag = plt.plot(label_x_t, yvals, 'g', label='polyfit tag train values')

    if len(label_x_q) > 0:
        # z2 = np.polyfit(label_x_q, label_y_q, 2)  # 用3次多项式拟合
        # p2 = np.poly1d(z2)
        # print(p2)  # 在屏幕上打印拟合多项式
        # yvals = p2(label_x_q)  # 也可以使用yvals=np.polyval(z1,x)
        # plot3 =
        plt.plot(label_x_q, label_y_q, '*', label='original query values')
        # plot4 = plt.plot(label_x_q, yvals, 'b', label='polyfit query values')

    plt.xlabel('node distance')
    plt.ylabel('node energy')
    plt.legend(loc=4)
    plt.title('polyfitting id:: ' + str(item_id))

    plt.show()


if __name__ == "__main__":
    data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\"
    train_file, query_file, points_num = data_path + "sift_train.bin", data_path + "sift_query.bin", 1000000
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    data_list = []
    for i in range(0, points_num):
        if i%10000 == 0:
            print("tags is %d" % i)
        data_t = train.read(4)
        data_num_t = struct.unpack("I", data_t)[0]

        data_q = query.read(4)
        data_num_q = struct.unpack("I", data_q)[0]

        label_x_t = np.zeros(data_num_t)
        label_y_t = np.zeros(data_num_t)

        label_x_q = np.zeros(data_num_q)
        label_y_q = np.zeros(data_num_q)

        tag_t = np.full(data_num_t, i)
        tag_q = np.full(data_num_q, i)

        new_data_num_t = data_num_t

        for j in range(0, data_num_t):
            data = train.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x_t[j] = data_float
            data = train.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y_t[j] = data_float

            if label_y_t[j] == 0:
                new_data_num_t -= 1

        new_label_x_t = np.zeros(new_data_num_t)
        new_label_y_t = np.zeros(new_data_num_t)

        cur = 0
        for j in range(0, data_num_t):
            if label_y_t[j] == 0:
                continue
            new_label_x_t[cur] = label_x_t[j]
            new_label_y_t[cur] = label_y_t[j]
            cur += 1

        label_x_t = new_label_x_t
        label_y_t = new_label_y_t

        for j in range(0, data_num_q):
            data = query.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x_q[j] = data_float
            data = query.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y_q[j] = data_float

        data_list.append((label_x_t, label_y_t, label_x_q, label_y_q))

    num = int(input())
    print(points_num)
    while num > 0:
        num -= 1
        item_id = int(input())
        curve_line(data_list[item_id][0], data_list[item_id][1], data_list[item_id][2], data_list[item_id][3],item_id)

#
# 1 987636
# 124761 889819 532108 987074 987215 20845 517238 833054 987636
#
