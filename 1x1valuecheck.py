import numpy as np
import matplotlib as plt
import csv
import struct
from sys import argv

if __name__ == "__main__":
    argv.pop(0)
    line_file_name = argv[0]
    query_file_name = argv[1]

    data_path = "D:\\DATA\\extra_data\\energy_data\\"
    query = open(query_file_name, "rb")
    lines = open(line_file_name, "rb")

    data_t = lines.read(4)
    points_num = struct.unpack(data_t,"I")
    data_t = lines.read(4)
    line_dim = struct.unpack(data_t,"I")

    node_energy = np.zeros(points_num)

    for i in range(0,points_num):
        data_t = lines.read(4)



