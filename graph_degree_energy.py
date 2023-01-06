import numpy as np
import csv
from sys import argv
import struct
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import random
import matplotlib as mlib
import matplotlib.pyplot as plt


def standardization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_degree_energy_relation(degree_file, energy_file, cluster_file):
    degree = open(degree_file, "rb")
    energy = open(energy_file, "rb")
    cluster = open(cluster_file, "rb")
    item = cluster.read(4)
    cluster_count = struct.unpack("I", item)[0]
    item = degree.read(4)
    points_num = struct.unpack("I", item)[0]
    belong = {}
    cluster_degree_size = np.zeros(cluster_count)
    cluster_energy_size = np.zeros(cluster_count)
    cluster_ave_degree = np.zeros(cluster_count)
    cluster_ave_energy = np.zeros(cluster_count)
    cluster_max_energy = np.zeros(cluster_count)

    for i in range(0, points_num):
        item = cluster.read(4)
        node_cluster = struct.unpack("I", item)[0]
        belong[i] = node_cluster
        if i % 100000 == 0:
            print("energy cluster:: is" + str(belong[i]) + "count is:: " + str(i))

    for i in range(0, points_num):
        item = degree.read(4)
        node_degree = struct.unpack("I", item)[0]
        cluster_ave_degree[belong[i]] += node_degree
        cluster_degree_size[belong[i]] += 1.0

    for i in range(0, points_num):
        item = energy.read(4)
        size = struct.unpack("I", item)[0]
        for j in range(0, size):
            item = energy.read(4)
            depth = struct.unpack("I", item)[0]
            item = energy.read(4)
            node_dist = struct.unpack("f", item)[0]
            item = energy.read(4)
            node_energy = struct.unpack("f", item)[0]
            cluster_ave_energy[belong[i]] += node_energy - node_dist
            cluster_max_energy[belong[i]] = max(cluster_max_energy[belong[i]], node_energy - node_dist)
            cluster_energy_size[belong[i]] += 1.0

    for i in range(0, cluster_count):
        if cluster_energy_size[i] != 0:
            cluster_ave_energy[i] /= cluster_energy_size[i]
        if cluster_degree_size[i] != 0:
            cluster_ave_degree[i] /= cluster_degree_size[i]
        print(cluster_ave_energy[i])

    limited = 64
    base = np.arange(0, limited)

    cluster_ave_energy = standardization(cluster_ave_energy)
    cluster_ave_degree = standardization(cluster_ave_degree)
    cluster_energy_size = standardization(cluster_energy_size)
    cluster_degree_size = standardization(cluster_degree_size)

    cluster_ave_degree *= -1
    cluster_ave_degree += 1
    plt.bar(base, cluster_ave_energy[:limited], label='ave_energy')
    plt.bar(base, cluster_ave_degree[:limited], label='ave_degree')
    #plt.bar(base, cluster_energy_size[:limited], label='energy_size')

    # plt.plot(base, cluster_ave_energy[:limited], c="b",label = 'ave_energy')
    # plt.plot(base, cluster_ave_degree[:limited], c="y",label = 'ave_degree')
    # plt.plot(base, cluster_energy_size[:limited], c='r',label = 'energy_size')
    # plt.plot(base, cluster_degree_size[:limited], c='g')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    argv.pop(0)
    degree_file_name = argv[0]

    energy_file_name = argv[1]
    cluster_file = argv[2]
    energy_data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\original_energy\\"
    graph_data_path = "D:\\DATA\\extra_data\\energy_data\\sift_energy\\graph_feature\\"
    get_degree_energy_relation(graph_data_path + degree_file_name, energy_data_path + energy_file_name,
                               energy_data_path + cluster_file)

#
# sift_nsg_degree.bin sift_query_path.bin sift_cluster1000.bin
# gist_nsg_degree.bin gist_query_path.bin gist_cluster256.bin
#
