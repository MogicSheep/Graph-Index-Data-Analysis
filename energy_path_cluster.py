import numpy as np
import csv
from sys import argv
import struct
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import random
import matplotlib as mlib
import matplotlib.pyplot as plt


def data_3d_analysis(filename, cluster_file, points_num):
    fin = open(filename, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    data_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in data_list.values():
            data_list[node_cluster] = []

    for i in range(0, points_num):
        item = fin.read(4)
        size = struct.unpack("I", item)[0]

        for j in range(0, size):
            item = fin.read(4)
            depth = struct.unpack("I", item)[0]
            item = fin.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = fin.read(4)
            node_energy = struct.unpack("f", item)[0]
            data_list[belong[i]].append((depth, node_dist, node_energy))

    for cluster_id, data_item in data_list.items():
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(data_item)
        cube = np.zeros(size, 3)
        for j in range(0, size):
            cube[j][0] = data_item[j][0]
            cube[j][1] = data_item[j][1]
            cube[j][2] = data_item[j][2]
        ax.scatter(cube[:0], cube[:1], cube[:2], marker='o')
        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        plt.legend(loc='upper right')
        plt.show()


def data_3d_analysis_up_hull(filename, cluster_file, points_num):
    fin = open(filename, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    data_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in data_list.values():
            data_list[node_cluster] = []

    for i in range(0, points_num):
        item = fin.read(4)
        size = struct.unpack("I", item)[0]

        for j in range(0, size):
            item = fin.read(4)
            depth = struct.unpack("I", item)[0]
            item = fin.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = fin.read(4)
            node_energy = struct.unpack("f", item)[0]
            data_list[belong[i]].append((depth, node_dist, node_energy))

    for cluster_id, data_item in data_list.items():
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(data_item)
        cube = np.zeros((size, 3))
        for j in range(0, size):
            cube[j][0] = data_item[j][0]
            cube[j][1] = data_item[j][1]
            cube[j][2] = data_item[j][2]
        hull = ConvexHull(cube)
        for vert in hull.vertices:
            ax.scatter(cube[vert, 0], cube[vert, 1], cube[vert, 2], marker='o')
        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        ax.legend(loc='upper right')
        plt.show()


def hybrid_data_analysis_3d(train_file, query_file, cluster_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    train_list = {}
    query_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in train_list.values():
            train_list[node_cluster] = []
        if node_cluster not in query_list.values():
            query_list[node_cluster] = []

    for i in range(0, points_num):
        item = train.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = train.read(4)
            depth = struct.unpack("I", item)[0]
            item = train.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = train.read(4)
            node_energy = struct.unpack("f", item)[0]
            train_list[belong[i]].append((depth, node_dist, node_energy))

    for i in range(0, points_num):
        item = query.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = query.read(4)
            depth = struct.unpack("I", item)[0]
            item = query.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = query.read(4)
            node_energy = struct.unpack("f", item)[0]
            query_list[belong[i]].append((depth, node_dist, node_energy))

    for cluster_id in range(0, cluster_count):
        train_item = train_list[cluster_id]
        query_item = query_list[cluster_id]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(train_item)
        cube = np.zeros((size, 3))
        for j in range(0, size):
            cube[j][0] = train_item[j][0]
            cube[j][1] = train_item[j][1]
            cube[j][2] = train_item[j][2]
        ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], marker='o', c='b', alpha=1 / 20, label='train data')

        size = len(query_item)
        cube = np.zeros((size, 3))
        for j in range(0, size):
            cube[j][0] = query_item[j][0]
            cube[j][1] = query_item[j][1]
            cube[j][2] = query_item[j][2]
        ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], marker='o', c='y', label='query data')
        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        ax.legend(loc='best')
        plt.show()


def hybrid_hull_analysis_3d(train_file, query_file, cluster_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    train_list = {}
    query_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in train_list.values():
            train_list[node_cluster] = []
        if node_cluster not in query_list.values():
            query_list[node_cluster] = []

    for i in range(0, points_num):
        item = train.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = train.read(4)
            depth = struct.unpack("I", item)[0]
            item = train.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = train.read(4)
            node_energy = struct.unpack("f", item)[0]
            train_list[belong[i]].append((depth, node_dist, node_energy))

    for i in range(0, points_num):
        item = query.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = query.read(4)
            depth = struct.unpack("I", item)[0]
            item = query.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = query.read(4)
            node_energy = struct.unpack("f", item)[0]
            query_list[belong[i]].append((depth, node_dist, node_energy))

    for cluster_id in range(0, cluster_count):
        train_item = train_list[cluster_id]
        query_item = query_list[cluster_id]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(train_item)
        cube = np.zeros((size, 3))
        for j in range(0, size):
            cube[j][0] = train_item[j][0]
            cube[j][1] = train_item[j][1]
            cube[j][2] = train_item[j][2]
        hull = ConvexHull(cube)
        for vert in hull.vertices:
            ax.scatter(cube[vert, 0], cube[vert, 1], cube[vert, 2], marker='o', c='b', label='train hull node')

        size = len(query_item)
        cube = np.zeros((size, 3))
        for j in range(0, size):
            cube[j][0] = query_item[j][0]
            cube[j][1] = query_item[j][1]
            cube[j][2] = query_item[j][2]
        hull = ConvexHull(cube)
        for vert in hull.vertices:
            ax.scatter(cube[vert, 0], cube[vert, 1], cube[vert, 2], marker='o', c='y', label='query hull node')
        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        plt.legend(loc='best')
        plt.show()


def hybrid_line_analysis_3d(train_file, query_file, cluster_file, points_num, line_size):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    train_list = {}
    query_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in train_list.values():
            train_list[node_cluster] = []
        if node_cluster not in query_list.values():
            query_list[node_cluster] = []

    for i in range(0, points_num):
        item = train.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = train.read(4)
            depth = struct.unpack("I", item)[0]
            item = train.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = train.read(4)
            node_energy = struct.unpack("f", item)[0]
            train_list[belong[i]].append((depth, node_dist, node_energy))

    for i in range(0, points_num):
        item = query.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = query.read(4)
            depth = struct.unpack("I", item)[0]
            item = query.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = query.read(4)
            node_energy = struct.unpack("f", item)[0]
            query_list[belong[i]].append((depth, node_dist, node_energy))
    bad_line = 0
    all_line = 0
    for cluster_id in range(0, cluster_count):
        train_item = train_list[cluster_id]
        query_item = query_list[cluster_id]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(train_item)
        cube = np.zeros((size, 3))
        line_train = np.zeros(line_size)
        right_upper = -1e9
        for j in range(0, size):
            cube[j][0] = train_item[j][0]
            cube[j][1] = train_item[j][1]
            cube[j][2] = train_item[j][2]
            right_upper = max(right_upper, cube[j][1])
            if cube[j][0] < line_size:
                line_train[int(cube[j][0])] = max(line_train[int(cube[j][0])], cube[j][2] - cube[j][1])
        max_line_energy = line_train[line_size - 1]
        for j in range(line_size - 1, -1, -1):
            max_line_energy = max(max_line_energy, line_train[j])
            line_train[j] = max_line_energy
        for j in range(0, line_size):
            ax.plot([j, j], [0, right_upper], [line_train[j], right_upper + line_train[j]], label='train curve', c='b')

        size = len(query_item)
        cube = np.zeros((size, 3))
        line_query = np.zeros(line_size)
        right_upper = -1e9
        for j in range(0, size):
            cube[j][0] = query_item[j][0]
            cube[j][1] = query_item[j][1]
            cube[j][2] = query_item[j][2]
            right_upper = max(right_upper, cube[j][1])
            if cube[j][0] < line_size:
                line_query[int(cube[j][0])] = max(line_query[int(cube[j][0])], cube[j][2] - cube[j][1])

        for j in range(0, line_size):
            if line_query[j] == 0:
                continue
            if line_query[j] > line_train[j]:
                color = 'r'
                bad_line += 1
            else:
                color = 'y'
            ax.plot([j, j], [0, right_upper], [line_query[j], right_upper + line_query[j]], label='query curve',
                    c=color)

        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        plt.legend(loc='best')
        plt.show()

    print(bad_line)


def hybrid_curve_analysis_3d(train_file, query_file, cluster_file, points_num):
    train = open(train_file, "rb")
    query = open(query_file, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    belong = {}
    train_list = {}
    query_list = {}
    print("cluster count:: " + str(cluster_count))
    for i in range(0, points_num):
        data_t = cluster.read(4)
        node_cluster = struct.unpack("i", data_t)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))
        if node_cluster not in train_list.values():
            train_list[node_cluster] = []
        if node_cluster not in query_list.values():
            query_list[node_cluster] = []

    for i in range(0, points_num):
        item = train.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = train.read(4)
            depth = struct.unpack("I", item)[0]
            item = train.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = train.read(4)
            node_energy = struct.unpack("f", item)[0]
            train_list[belong[i]].append((depth, node_dist, node_energy))

    for i in range(0, points_num):
        item = query.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = query.read(4)
            depth = struct.unpack("I", item)[0]
            item = query.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = query.read(4)
            node_energy = struct.unpack("f", item)[0]
            query_list[belong[i]].append((depth, node_dist, node_energy))

    for cluster_id in range(0, cluster_count):
        train_item = train_list[cluster_id]
        query_item = query_list[cluster_id]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        size = len(train_item)
        cube = np.zeros((size, 3))

        train_item.sort(reverse=False)
        query_item.sort(reverse=False)
        pre = 0
        for j in range(0, size):
            cube[j][0] = train_item[j][0]
            cube[j][1] = train_item[j][1]
            cube[j][2] = train_item[j][2]
            if j > 0 and cube[j][0] != cube[j - 1][0] or j == size - 1:
                ax.plot(cube[pre:j, 0], cube[pre:j, 1], cube[pre:j, 2], label='train curve', c='b')
                pre = j

        size = len(query_item)
        cube = np.zeros((size, 3))
        pre = 0
        for j in range(0, size):
            cube[j][0] = query_item[j][0]
            cube[j][1] = query_item[j][1]
            cube[j][2] = query_item[j][2]
            if j > 0 and cube[j][0] != cube[j - 1][0] or j == size - 1:
                ax.plot(cube[pre:j, 0], cube[pre:j, 1], cube[pre:j, 2], label='query curve', c='y')
                pre = j

        ax.set_xlabel('depth')
        ax.set_ylabel('X node distance')
        ax.set_zlabel('Y node energy')
        ax.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    argv.pop(0)
    train_file_name = argv[0]
    query_file_name = argv[1]
    dataset = train_file_name.split('_')[0]
    plt.legend(loc='upper right')
    cluster_file = argv[2]
    data_path = "D:\\DATA\\extra_data\\energy_data\\" + dataset + "_energy\\original_energy\\"
    hybrid_data_analysis_3d(data_path + train_file_name, data_path + query_file_name, data_path + cluster_file,
                            1000000)
    # data_3d_analysis(data_path + train_file_name, data_path + cluster_file, 1000000)
    print("analysis begin")

# ###
# sift_train_path.bin sift_query_path.bin sift_cluster2000.bin
# gist_train_path.bin gist_query_path.bin gist_cluster256.bin
# sift_train_pool.bin sift_query_pool.bin sift_cluster2000.bin
# gist_train_pool.bin gist_query_pool.bin gist_cluster256.bin
# ###
