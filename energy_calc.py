import numpy as np
import csv
import struct
from scipy.optimize import curve_fit
import matplotlib as mlib
import matplotlib.pyplot as plt


def draw_line_3d(file_name):
    mlib.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlabel("node_id")
    plt.ylabel("node_distance")
    points_num = 100
    f = open(file_name, "rb")

    for i in range(0, points_num):
        data = f.read(4)
        data_num = struct.unpack("I", data)[0]

        label_x = np.zeros(data_num)
        label_y = np.zeros(data_num)
        tag = np.full(data_num, i)
        print(data_num)
        for j in range(0, data_num):
            data = f.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x[j] = data_float
            data = f.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y[j] = data_float

            print(label_x[j])
            print(label_y[j])

        ax.plot(tag, label_x, label_y)

    ax.legend()
    plt.show()


def line_cur(x, y):
    z1 = np.polyfit(x, y, 2)  # 用3次多项式拟合
    p1 = np.poly1d(z1)
    print(p1)  # 在屏幕上打印拟合多项式
    yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)
    plt.title('polyfitting')
    plt.show()


def data_line_cur(file_name, points_num):
    f = open(file_name, "rb")

    for i in range(0, points_num):
        data = f.read(4)
        data_num = struct.unpack("I", data)[0]

        label_x = np.zeros(data_num)
        label_y = np.zeros(data_num)
        tag = np.full(data_num, i)
        print(data_num)
        for j in range(0, data_num):
            data = f.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x[j] = data_float
            data = f.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y[j] = data_float

        if data_num > 10:
            line_cur(label_x, label_y)


def generate_new_line(func_lin, train_x, train_y, rate):
    new_size = train_x.size * rate
    l = 0
    r = 200000
    ans = r
    while l <= r:
        mid = (l + r) / 2
        cur = 0
        for i in range(0,train_x.size):
            a = 0
            x = train_x[i]
            y = train_y[i]
            a += func_lin[0] * x * x + func_lin[1] * x + func_lin[2]
            if a + mid > y:
                cur += 1

        if cur >= new_size:
            ans = mid
            r = mid - 1
        else:
            l = mid + 1
    return ans


def train_query_check(train_file, query_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    for i in range(0, points_num):
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
        print(data_num_t)
        print(data_num_q)

        for j in range(0, data_num_t):
            data = train.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x_t[j] = data_float
            data = train.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y_t[j] = data_float

        for j in range(0, data_num_q):
            data = query.read(4)
            data_float = struct.unpack("f", data)[0]
            label_x_q[j] = data_float
            data = query.read(4)
            data_float = struct.unpack("f", data)[0]
            label_y_q[j] = data_float

        if data_num_t > 5 and data_num_q > 3:
            z1 = np.polyfit(label_x_t, label_y_t, 2)  # 用3次多项式拟合
            p1 = np.poly1d(z1)
            tag = generate_new_line(z1, label_x_t, label_y_t, 0.95)
            print(p1)  # 在屏幕上打印拟合多项式
            yvals = p1(label_x_t)  # 也可以使用yvals=np.polyval(z1,x)

            plot1 = plt.plot(label_x_t, label_y_t, '*', label='original train values')
            plot2 = plt.plot(label_x_t, yvals, 'r', label='polyfit train values')

            z1[2] += tag
            p1 = np.poly1d(z1)
            yvals = p1(label_x_t)

            plottag = plt.plot(label_x_t, yvals, 'g', label='polyfit tag train values')

            z2 = np.polyfit(label_x_q, label_y_q, 2)  # 用3次多项式拟合
            p2 = np.poly1d(z2)
            print(p2)  # 在屏幕上打印拟合多项式
            yvals = p2(label_x_q)  # 也可以使用yvals=np.polyval(z1,x)
            plot3 = plt.plot(label_x_q, label_y_q, '*', label='original query values')
            plot4 = plt.plot(label_x_q, yvals, 'b', label='polyfit query values')

            plt.xlabel('node distance')
            plt.ylabel('node energy')
            plt.legend(loc=4)
            plt.title('polyfitting')

            plt.show()


if __name__ == "__main__":
    train_query_check("energy_all.bin", "energy_query.bin", 1000000)
    data_line_cur("energy_line.bin", 100)
    data_line_cur("energy_query.bin", 100)
    draw_line_3d("energy_line.bin")
    draw_line_3d("energy_query.bin")
