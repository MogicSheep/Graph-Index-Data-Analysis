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
    l = -300000
    r = 300000
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
            r = mid - 0.001
        else:
            l = mid + 0.001
    return ans


def line_curve(X, Y):
    z1 = np.polyfit(X, Y, 2)  # 用3次多项式拟合
    tag = generate_new_line(z1, X, Y, 0.99)
    z1[2] += tag
    return z1


def random_line_curve(data_flat_list, data_type):
    count = 50000
    count = min(count,len(data_flat_list))
    data_random_list = random.sample(data_flat_list, count)
    data_random_list.sort()

    tag_y = -1
    up_curve = []
    for item in data_random_list:
        if (item[1] > tag_y):
            up_curve.append(item)
            tag_y = item[1]
    count = len(up_curve)


    nodes_array_x = np.zeros(count)
    nodes_array_y = np.zeros(count)
    print(count)
    for i in range(0, count):
        nodes_array_x[i] = up_curve[i][0]
        nodes_array_y[i] = up_curve[i][1]

    general = line_curve(nodes_array_x, nodes_array_y)

    count = len(data_random_list)

    nodes_array_x = np.zeros(count)
    nodes_array_y = np.zeros(count)
    print(count)
    for i in range(0, count):
        nodes_array_x[i] = data_random_list[i][0]
        nodes_array_y[i] = data_random_list[i][1]

    p1 = np.poly1d(general)
    yvals = p1(nodes_array_x)

    if data_type == "train":
        plt.xlabel('node distance')
        plt.ylabel('node energy')
        plt.title("node energy of sift_learn.fvecs")
        plt.plot(nodes_array_x, yvals, 'r', label='polyfit train values')
        plt.plot(nodes_array_x, nodes_array_y, '*', label='original train values')
        plt.show()
    else:
        plt.xlabel('node distance')
        plt.ylabel('node energy')
        plt.title("node energy of sift_query.fvecs")
        plt.plot(nodes_array_x, yvals, 'b', label='polyfit query values')
        plt.plot(nodes_array_x, nodes_array_y, 'o', label='original query values')
        plt.legend()
        plt.show()
        exit(1)
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
    belong = {}
    cluster_count = struct.unpack("I",data_cur)[0]
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i%100000 == 0:
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
        node_points = []
        for j in range(0, data_num_t):
            data = train.read(4)
            data_x = struct.unpack("f", data)[0]
            data = train.read(4)
            data_y = struct.unpack("f", data)[0]
            data_flat_list.append((data_x, data_y))
            data_list[belong[i]].append((data_x, data_y))

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
            data_query_list[belong[i]].append((data_x,data_y))


    query_board_line = random_line_curve(data_flat_list, "query")
    plt.show()
    count = 0
    use_count = 0
    save_energy.write(struct.pack("I", cluster_count))
    save_energy.write(struct.pack("f",general_line[2]))
    save_energy.write(struct.pack("f", general_line[1]))
    save_energy.write(struct.pack("f", general_line[0]))

    for cluster_id, node_list in data_list.items():
        node_x = np.zeros(len(node_list))
        node_y = np.zeros(len(node_list))
        query_list = data_query_list[cluster_id]
        node_x_q = np.zeros(len(query_list))
        node_y_q = np.zeros(len(query_list))

        line_A = general_line[0]
        line_B = general_line[1]
        line_C = general_line[2]

        if len(node_list) > 3:
            use_count += 1
            for j in range(0, len(node_list)):
                node_x[j] = node_list[j][0]
                node_y[j] = node_list[j][1]

            for j in range(0,len(query_list)):
                node_x_q[j] = query_list[j][0]
                node_y_q[j] = query_list[j][1]

            plt.title('polyfitting clsuter ' + str(cluster_id))
            cluster_line = random_line_curve(node_list,"train")
            # cluster_line = random_line_curve(query_list,"query")
            line_A = cluster_line[2]
            line_B = cluster_line[1]
            line_C = cluster_line[0]
            p1 = np.poly1d(cluster_line)
            # yvals = p1(node_x_q)
            # if cluster_id == 110:
            #     plt.xlabel('node distance')
            #     plt.ylabel('node energy')
            #     plt.legend(loc=4)
            #     plt.title('polyfitting before')
            #     plt.show()
            #
            #     random_line_curve(node_list, "train")
            #     #plt.plot(node_x_q, node_y_q, 'o', label='original query values')
            #     plt.xlabel('node distance')
            #     plt.ylabel('node energy')
            #     plt.legend(loc=4)
            #     plt.title('polyfitting clsuter ' + str(cluster_id))
            #     plt.show()
            # for j in range(0,len(query_list)):
            #     if yvals[j] < node_y_q[j]:
            #         print("bad case")
            #         print("cluster_id" + str(cluster_id))
            #         random_line_curve(node_list, "train")
            #         plt.plot(node_x,node_y,'*',label = 'original train values')
            #         plt.plot(node_x_q, node_y_q, 'o', label='original query values')
            #         plt.show()
            #         break

        id = struct.pack("I", int(cluster_id))
        print(cluster_id)
        line_A = struct.pack("f", line_A)
        line_B = struct.pack("f", line_B)
        line_C = struct.pack("f", line_C)
        save_energy.write(id)
        save_energy.write(line_A)
        save_energy.write(line_B)
        save_energy.write(line_C)
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
    data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\original_energy\\"
    print("test begin")
    general_line, data_list = cluster_energy_line_cure(data_path + train_file_name, data_path + query_file_name,
                                                       data_path + save_file_name, data_path + cluster_file_name,
                                                       1000000)


    # train_query_check(data_path + train_file_name, data_path + query_file_name, data_path + save_file_name, 1000000)
    # data_line_cur("energy_line.bin", 100)
    # draw_line_3d("energy_line.bin")
    # draw_line_3d("energy_query.bin")
