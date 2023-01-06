import numpy as np
import csv
from sys import argv
import struct
from scipy.optimize import curve_fit
import random
import matplotlib as mlib
import matplotlib.pyplot as plt


def generate_new_line(func_lin, train_x, train_y, rate):
    new_size = train_x.size * rate
    l = -500000
    r = 5000000
    ans = r
    while l <= r:
        mid = (l + r) / 2.0
        cur = 0
        for i in range(0, train_x.size):
            a = 0
            x = train_x[i]
            y = train_y[i]
            a += func_lin[0] * x + func_lin[1]
            if a + mid > y:
                cur += 1

        if cur >= new_size:
            ans = mid
            r = mid - 0.001
        else:
            l = mid + 0.001
    return ans


def line_curve(X, Y):
    z1 = np.poly1d([1, 0])
    tag = generate_new_line(z1, X, Y, 0.9999)
    z1[1] += tag
    return z1


def random_line_curve(data_flat_list, data_type):
    # data_flat_list.sort()
    tag_y = -1
    up_curve = []
    for item in data_flat_list:
        if (item[1] > tag_y):
            up_curve.append(item)
            tag_y = item[1]
    count = 50000

    nodes_array_x = np.zeros(count)
    nodes_array_y = np.zeros(count)
    print(len(data_flat_list))
    for i in range(0, count):
        nodes_array_x[i] = data_flat_list[i][0]
        nodes_array_y[i] = data_flat_list[i][1]

    general = line_curve(nodes_array_x, nodes_array_y)
    p1 = np.poly1d(general)
    yvals = p1(nodes_array_x)
    if data_type == "train":
        plt.plot(nodes_array_x, yvals, 'r', label='polyfit train line')
        plt.plot(nodes_array_x, nodes_array_y, '*', label='original train values')
        plt.show()
    elif data_type == "query":
        plt.plot(nodes_array_x, yvals, 'b', label='polyfit query line')
        plt.plot(nodes_array_x, nodes_array_y, 'o', label='original query values')
        plt.show()
    elif data_type == "extra":
        print("extra tag")
        print(general[0])
        print(general[1])
        plt.plot(nodes_array_x, yvals, 'g', label='polyfit extra line')
        plt.plot(nodes_array_x, nodes_array_y, '^', label='original extra values')
        plt.show()
    return general


def cluster_energy_line_cure(train_file, query_file, save_file, cluster_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")
    data_cur = cluster.read(4)
    save_energy = open(save_file, "wb")
    print("test all begin")
    all_points_num = 0
    data_list = {}
    data_query_list = {}
    data_flat_list = []
    belong = np.zeros(points_num)
    cluster_count = struct.unpack("I", data_cur)[0]
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in data_list.values():
            data_list[node_cluster] = []
        if node_cluster not in data_query_list.values():
            data_query_list[node_cluster] = []

    for i in range(0, points_num):
        if i % 100000 == 0:
            print("now train iterater:: is " + str(i))
        data_t = train.read(4)
        data_num_t = struct.unpack("I", data_t)[0]
        all_points_num += data_num_t
        for j in range(0, data_num_t):
            data = train.read(4)
            data_x = struct.unpack("f", data)[0]
            data = train.read(4)
            data_y = struct.unpack("f", data)[0]
            data_flat_list.append((data_x, data_y))
            data_list[belong[i]].append((data_x, data_y))

    general_line = random_line_curve(data_flat_list, "train")

    exit(1)
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
            data_query_list[belong[i]].append((data_x, data_y))

    # query_board_line = random_line_curve(data_flat_list, "query")
    # plt.show()
    count = 0
    use_count = 0
    save_energy.write(struct.pack("I", cluster_count))
    save_energy.write(struct.pack("f", general_line[0]))
    save_energy.write(struct.pack("f", general_line[1]))

    for cluster_id, node_list in data_list.items():
        node_x = np.zeros(len(node_list))
        node_y = np.zeros(len(node_list))
        query_list = data_query_list[cluster_id]
        node_x_q = np.zeros(len(query_list))
        node_y_q = np.zeros(len(query_list))

        line_A = general_line[0]
        line_B = general_line[1]

        if len(node_list) > 30:
            use_count += 1
            for j in range(0, len(node_list)):
                node_x[j] = node_list[j][0]
                node_y[j] = node_list[j][1]

            for j in range(0, len(query_list)):
                node_x_q[j] = query_list[j][0]
                node_y_q[j] = query_list[j][1]

            cluster_line = random_line_curve(node_list, "train")
            p1 = np.poly1d(cluster_line)
            print(cluster_id)
            print(p1)
            plt.xlabel('node distance')
            plt.ylabel('node energy')
            plt.title('polyfitting clsuter ' + str(cluster_id))
            plt.show()
            if len(query_list) > 3:
                cluster_line = random_line_curve(query_list, "query")

                line_A = cluster_line[0]
                line_B = cluster_line[1]
                p1 = np.poly1d(cluster_line)
                yvals = p1(node_x_q)
                for j in range(0, len(query_list)):
                    if yvals[j] < node_y_q[j]:
                        print("bad case")
                        print("cluster_id" + str(cluster_id))
                        random_line_curve(node_list, "query")
                        plt.xlabel('node distance')
                        plt.ylabel('node energy')
                        plt.title('polyfitting clsuter ' + str(cluster_id))
                        plt.plot(node_x, node_y, '*', label='original train values')
                        plt.plot(node_x_q, node_y_q, 'o', label='original query values')
                        plt.show()
                    break

        id = struct.pack("I", int(cluster_id))
        # print(cluster_id)
        line_A = struct.pack("f", line_A)
        line_B = struct.pack("f", line_B)
        save_energy.write(id)
        save_energy.write(line_A)
        save_energy.write(line_B)
        if count % 100 == 0:
            print("now count iterate:: is " + str(count))
        count += 1
    print("useful nodes is :: " + str(use_count))
    plt.show()
    return general_line, data_list


if __name__ == "__main__":
    argv.pop(0)
    train_file_name = argv[0]
    query_file_name = argv[1]
    cluster_file_name = argv[2]
    save_file_name = argv[3]
    data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\"
    print("test begin")
    general_line, data_list = cluster_energy_line_cure(data_path + train_file_name, data_path + query_file_name,
                                                       data_path + save_file_name, data_path + cluster_file_name,
                                                       1000000)

    # train_query_check(data_path + train_file_name, data_path + query_file_name, data_path + save_file_name, 1000000)
    # data_line_cur("energy_line.bin", 100)
    # draw_line_3d("energy_line.bin")
    # draw_line_3d("energy_query.bin")
