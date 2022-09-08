import numpy as np
import pandas as pd
import struct
import csv

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32').astype(np.float32), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# put the part of file into cache, prevent the slow load that file is too big
def fvecs_read_mmap(fname):
    x = np.memmap(fname, dtype='int32', mode='r', order='C')
    # x = np.memmap(fname, dtype='int32')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:], d


def generate_turb_vec_surface(nq, d):
    norm = np.random.normal
    U = norm(size=(d, nq))  # 随机生成N个d维空间中的坐标
    dev = np.sqrt(np.sum(U ** 2, axis=0))
    X = U / dev  # 对这N个坐标进行单位化，得到d维球面上的N个样本点
    return X


def generate_turb_vec_inside(nq, d):
    dim = d  # 空间维度
    norm = np.random.normal
    U = norm(size=(dim, nq))  # 随机生成d维空间中的N个坐标点
    dev = np.sqrt(np.sum(U ** 2, axis=0))
    radius = np.power(np.random.random(size=(1, nq)), 1 / dim)  # 对N个坐标点分别随机生成一个半径
    X = np.multiply(radius, U) / dev
    return X


def load_nn_distance(filename):
    df = pd.read_csv("static_energy.csv")



if __name__ == "__main__":
    filename = "E:\\DATA\\vector_data\\sift1m"
    xq, d = fvecs_read(filename + "\\sift_query.fvecs")
    nq, d = xq.shape

    ans_tag = np.zeros(nq, d)
    turb_surface = generate_turb_vec_surface(nq, d)
    turb_inside = generate_turb_vec_inside(nq, d)
    for i in range(0, nq):
        ans_tag[i] = xq[i] + turb_surface


    


