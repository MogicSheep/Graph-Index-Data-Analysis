import numpy as np
import csv
from sys import argv
import struct
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import random
import matplotlib as mlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


def virtual_voronoi_cell(train_file, query_file, cluster_file, item_count, pool_size, poly_count, save_file, show_plt):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")

    item = cluster.read(4)
    points_num = struct.unpack("I", item)[0]
    item = cluster.read(4)
    pyramid_size = struct.unpack("I", item)[0]

    cluster_pyramid = np.zeros((points_num, pyramid_size))
    node_energy_size = np.zeros(points_num)
    energy_func_result = np.zeros((points_num, poly_count))
    node_belong_to = {}
    train_list = {}
    query_list = {}
    print("node count:: " + str(points_num) + "\nlayer count:: " + str(pyramid_size))
    for i in range(0, points_num):
        for j in range(0, pyramid_size):
            item = cluster.read(4)
            cluster_id = struct.unpack("I", item)[0]
            cluster_pyramid[i][j] = cluster_id - 1

    print("cluster load finished")

    for i in range(0, points_num):
        item = train.read(4)
        size = struct.unpack("I", item)[0]
        train_list[i] = []
        node_energy_size[i] = size
        for j in range(0, size):
            item = train.read(4)
            depth = struct.unpack("I", item)[0]
            item = train.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = train.read(4)
            node_energy = struct.unpack("f", item)[0]
            train_list[i].append((depth, node_dist, node_energy))

    print("train data load finished")

    for i in range(0, points_num):
        item = query.read(4)
        size = struct.unpack("I", item)[0]
        query_list[i] = []
        node_energy_size[i] = size
        for j in range(0, size):
            item = query.read(4)
            depth = struct.unpack("I", item)[0]
            item = query.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = query.read(4)
            node_energy = struct.unpack("f", item)[0]
            query_list[i].append((depth, node_dist, node_energy))

    print("query data load finished")

    for j in range(0, pyramid_size):
        print("now layer count " + str(j))
        cluster_train_layer = {}
        cluster_query_layer = {}
        layer_func = {}
        for i in range(0, points_num):
            if cluster_pyramid[i][j] not in cluster_train_layer.keys():
                cluster_train_layer[cluster_pyramid[i][j]] = []
            cluster_train_layer[cluster_pyramid[i][j]].extend(train_list[i])

            if cluster_pyramid[i][j] not in cluster_query_layer.keys():
                cluster_query_layer[cluster_pyramid[i][j]] = []
            cluster_query_layer[cluster_pyramid[i][j]].extend(query_list[i])

        cluster_tqdm_layer = tqdm(cluster_train_layer.keys())
        for cluster_id in cluster_tqdm_layer:
            train_item = cluster_train_layer[cluster_id]
            query_item = cluster_query_layer[cluster_id]
            if len(train_item) < item_count:
                continue
            ax = plt.axes(projection='3d')
            size = len(train_item)
            cube = np.zeros((size, 3))
            train_item.sort(reverse=False)
            line_train = np.zeros(pool_size)
            right_upper = -1e9
            for k in range(0, size):
                cube[k][0] = train_item[k][0]
                cube[k][1] = train_item[k][1]
                cube[k][2] = train_item[k][2]
                right_upper = max(right_upper, cube[k][1])
                if cube[k][0] < pool_size:
                    line_train[int(cube[k][0])] = max(line_train[int(cube[k][0])], cube[k][2] - cube[k][1])

            if show_plt:
                for k in range(0, pool_size):
                    ax.plot([k, k], [0, right_upper], [line_train[k], right_upper + line_train[k]], label='train curve',
                            c='b')

            size = len(query_item)
            cube = np.zeros((size, 3))
            line_query = np.zeros(pool_size)
            right_upper = -1e9
            for k in range(0, size):
                cube[k][0] = query_item[k][0]
                cube[k][1] = query_item[k][1]
                cube[k][2] = query_item[k][2]
                right_upper = max(right_upper, cube[k][1])
                if cube[k][0] < pool_size:
                    line_query[int(cube[k][0])] = max(line_query[int(cube[k][0])], cube[k][2] - cube[k][1])

            if show_plt:
                for k in range(0, pool_size):
                    if line_query[k] == 0:
                        continue
                    if line_query[k] > line_train[k]:
                        color = 'r'
                    else:
                        color = 'y'
                    ax.plot([k, k], [0, right_upper], [line_query[k], right_upper + line_query[k]], label='query curve',
                            c=color)

                print("layer: " + str(j))
                print("clusterid: " + str(cluster_id))
                ax.set_title("original hybrid data")
                ax.set_xlabel('depth')
                ax.set_ylabel('X node distance')
                ax.set_zlabel('Y node energy')
                plt.show()

            pessimistic_line_train = np.zeros(pool_size)
            optimism_line_train = np.zeros(pool_size)
            original_line_train = np.zeros(pool_size)
            max_line_energy = line_train[pool_size - 1]
            for k in range(pool_size - 1, -1, -1):
                original_line_train[k] = line_train[k]
                max_line_energy = max(max_line_energy, line_train[k])
                optimism_line_train[k] = max_line_energy
                if line_train[k] < max_line_energy:
                    line_train[k] = 0

            pre_line_energy = max_line_energy
            for k in range(pool_size):
                pessimistic_line_train[k] = pre_line_energy
                if line_train[k] != 0:
                    pre_line_energy = line_train[k]

            func_base = np.arange(0, pool_size, 1)
            energy_func = np.polyfit(func_base, optimism_line_train, poly_count)
            if show_plt:
                plt.plot(func_base, original_line_train, '*', c='b', label='original data')
                plt.plot(func_base, line_query, 'o', c='y', label='query data')
                plt.plot(func_base, pessimistic_line_train, 'o', c='r', label='pessimistic', alpha=0.2)
                plt.plot(func_base, optimism_line_train, 'o', c='m', label='optimism', alpha=0.2)
                yvals = np.polyval(energy_func, func_base)
                plt.plot(func_base, yvals, c='g', label='energy curve')
                plt.legend(loc='upper right')
                plt.show()
            layer_func[cluster_id] = energy_func
        for i in range(0, points_num):
            if cluster_pyramid[i][j] in layer_func.keys():
                node_belong_to[i] = (j, cluster_pyramid[i][j])
                for k in range(0, poly_count):
                    energy_func_result[i][k] = layer_func[cluster_pyramid[i][j]][k]

    save_energy = open(save_file, "wb")
    save_energy.write(struct.pack("I", points_num))
    save_energy.write(struct.pack("I", poly_count))
    for i in range(0, points_num):
        for k in reversed(range(poly_count)):
            save_energy.write(struct.pack("f", energy_func_result[i][k]))

    cluster_dist = set()
    for node, belong_to in node_belong_to.items():
        cluster_dist.add(belong_to)

    print("useful node cluster count:: " + str(len(cluster_dist)))


if __name__ == "__main__":
    argv.pop(0)
    train_file_name = argv[0]
    query_file_name = argv[1]
    cluster_file = argv[2]
    save_file = argv[3]
    dataset = train_file_name.split('_')[0]
    print("data set:: " + dataset)
    cluster_path = "D:\\DATA\\extra_data\\energy_data\\" + dataset + "_energy\\graph_feature\\"
    data_path = "D:\\DATA\\extra_data\\energy_data\\" + dataset + "_energy\\original_energy\\"
    save_file_path = "D:\\DATA\\extra_data\\energy_data\\" + dataset + "_energy\\original_energy\\"

    virtual_voronoi_cell(data_path + train_file_name, data_path + query_file_name, cluster_path + cluster_file, 1000,
                         300, 3, save_file_path + save_file, False)

#
# sift_train_pool.bin sift_query_pool.bin virtual_vonoroi_256.bin vornoi_energy_func2.bin
# gist_train_pool.bin gist_query_pool.bin virtual_vonoroi_256.bin vornoi_energy_func2.bin
#
