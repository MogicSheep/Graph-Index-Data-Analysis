import numpy as np
import csv
from sys import argv
import struct
from scipy.optimize import curve_fit
import random
import matplotlib as mlib
import matplotlib.pyplot as plt

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d
def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32').astype(np.float32), d

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

        plt.plot(label_x, label_y, '^', label='query train values')

    plt.show()


def generate_new_line(func_lin, train_x, train_y, rate):
    new_size = train_x.size * rate
    l = 0
    r = 200000
    ans = r
    while l <= r:
        mid = (l + r) / 2
        cur = 0
        for i in range(0, train_x.size):
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


def train_query_check(train_file, query_file, save_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    save_energy = open(save_file, "wb")
    int_points_num = struct.pack("I", points_num)
    save_energy.write(int_points_num)
    useful_count = 0
    for i in range(0, points_num):
        if i % 10000 == 0:
            print("now iterater:: is " + str(i))
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
            if label_y_t[j] < 10:
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

        line_A = -1.00
        line_B = -1.00
        line_C = -1.00
        plot1 = plt.plot(label_x_t, label_y_t, '*', label='original train values')

        if new_data_num_t > 100:
            useful_count += 1
            # print(useful_count)
            z1 = np.polyfit(label_x_t, label_y_t, 2)  # 用3次多项式拟合
            p1 = np.poly1d(z1)
            tag = generate_new_line(z1, label_x_t, label_y_t, 0.9999)
            # print(p1)  # 在屏幕上打印拟合多项式
            yvals = p1(label_x_t)  # 也可以使用yvals=np.polyval(z1,x)

            plot1 = plt.plot(label_x_t, label_y_t, '*', label='original train values')
            plot2 = plt.plot(label_x_t, yvals, 'r', label='polyfit train values')

            z1[2] += tag
            # p1 = np.poly1d(z1)
            # yvals = p1(label_x_t)

            line_A = z1[0]
            line_B = z1[1]
            line_C = z1[2]

            plottag = plt.plot(label_x_t, yvals, 'g', label='polyfit tag train values')

            if data_num_q > 0:
                z2 = np.polyfit(label_x_q, label_y_q, 2)  # 用3次多项式拟合
                p2 = np.poly1d(z2)
                # print(p2)  # 在屏幕上打印拟合多项式
                yvals = p2(label_x_q)  # 也可以使用yvals=np.polyval(z1,x)
                plot3 = plt.plot(label_x_q, label_y_q, '*', label='original query values')
                # plot4 = plt.plot(label_x_q, yvals, 'b', label='polyfit query values')

            # plt.xlabel('node distance')
            # plt.ylabel('node energy')
            # plt.legend(loc=4)
            # plt.title('polyfitting id:: ' + str(i))

        id = struct.pack("I", i)
        line_A = struct.pack("f", line_A)
        line_B = struct.pack("f", line_B)
        line_C = struct.pack("f", line_C)
        save_energy.write(id)
        save_energy.write(line_A)
        save_energy.write(line_B)
        save_energy.write(line_C)

    plt.show()
    data_line_cur(data_path + query_file_name, 10000)
    print("count:: " + str(useful_count))


def line_curve(X, Y):
    z1 = np.polyfit(X, Y, 2)  # 用3次多项式拟合
    tag = generate_new_line(z1, X, Y, 1)
    z1[2] += tag
    return z1


def random_line_curve(data_flat_list, data_type):
    count = 40000
    data_random_list = random.sample(data_flat_list, count)

    data_random_list.sort()

    tag_y = -1
    up_curve = []
    for item in data_random_list:
        if (item[1] > tag_y):
            up_curve.append(item)
            tag_y = item[1]

    data_flat_list = up_curve
    count = len(up_curve)

    nodes_array_x = np.zeros(count)
    nodes_array_y = np.zeros(count)
    cur = 0
    for i in range(0, count):
        nodes_array_x[i] = data_flat_list[i][0]
        nodes_array_y[i] = data_flat_list[i][1]

    general = line_curve(nodes_array_x, nodes_array_y)
    p1 = np.poly1d(general)
    yvals = p1(nodes_array_x)
    if data_type == "train":
        plt.plot(nodes_array_x, yvals, 'r', label='polyfit train values')
        plt.plot(nodes_array_x, nodes_array_y, '*', label='original train values')
    else:
        plt.plot(nodes_array_x, yvals, 'b', label='polyfit query values')
        plt.plot(nodes_array_x, nodes_array_y, 'o', label='original query values')
    # plt.show()
    return general


def full_cover_energy_line_cure(train_file, query_file, save_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    save_energy = open(save_file, "wb")
    all_points_num = 0
    data_list = []
    data_flat_list = []
    for i in range(0, points_num):
        if i % 100000 == 0:
            print("now train iterater:: is " + str(i))
        data_t = train.read(4)
        data_num_t = struct.unpack("I", data_t)[0]

        all_points_num += data_num_t
        node_points = []
        for j in range(0, data_num_t):
            data = train.read(4)
            data_x = struct.unpack("f", data)[0]
            data = train.read(4)
            data_y = struct.unpack("f", data)[0]
            data_flat_list.append((data_x, data_y))
            if data_y > 10:
                node_points.append((data_x, data_y))

        data_list.append(node_points)

    general_line = random_line_curve(data_flat_list, "train")
    data_flat_list = []
    for i in range(0, points_num):
        if i % 100000 == 0:
            print("now query iterate:: is " + str(i))
        data_q = query.read(4)
        data_num_q = struct.unpack("I", data_q)[0]
        for j in range(0, data_num_q):
            data = query.read(4)
            data_x = struct.unpack("f", data)[0]
            data = query.read(4)
            data_y = struct.unpack("f", data)[0]
            data_flat_list.append((data_x, data_y))

    query_board_line = random_line_curve(data_flat_list, "query")

    plt.show()
    count = 0
    use_count = 0
    save_energy.write(struct.pack("I", points_num))
    for nodes in data_list:
        node_x = np.zeros(len(nodes))
        node_y = np.zeros(len(nodes))
        line_A = general_line[0]
        line_B = general_line[1]
        line_C = general_line[2]
        if len(nodes) > 10000000:
            use_count += 1
            for j in range(0, len(nodes)):
                node_x[j] = nodes[j][0]
                node_y[j] = nodes[j][1]

            z1 = np.polyfit(node_x, node_y, 2)  # 用3次多项式拟合
            tag = generate_new_line(z1, node_x, node_y, 0.9999)
            z1[2] += tag

            line_A = z1[0]
            line_B = z1[1]
            line_C = z1[2]
        id = struct.pack("I", count)
        line_A = struct.pack("f", line_A)
        line_B = struct.pack("f", line_B)
        line_C = struct.pack("f", line_C)
        save_energy.write(id)
        save_energy.write(line_A)
        save_energy.write(line_B)
        save_energy.write(line_C)
        if count % 100000 == 0:
            print("now count iterate:: is " + str(count))
        count += 1
    print("useful nodes is :: " + str(use_count))
    return general_line, data_list


if __name__ == "__main__":
    argv.pop(0)
    train_file_name = argv[0]
    query_file_name = argv[1]
    save_file_name = argv[2]
    data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\"
    general_line, data_list = full_cover_energy_line_cure(data_path + train_file_name, data_path + query_file_name,
                                                          data_path + save_file_name, 1000000)

    base_path = "D:\\DATA\\vector_data\\sift1m\\sift_base.fvecs"


    # train_query_check(data_path + train_file_name, data_path + query_file_name, data_path + save_file_name, 1000000)
    # data_line_cur("energy_line.bin", 100)
    # draw_line_3d("energy_line.bin")
    # draw_line_3d("energy_query.bin")
