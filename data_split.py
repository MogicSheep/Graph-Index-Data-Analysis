import numpy as np
import json
import random


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')







if __name__ == "__main__":
    np.random.seed(123)
    sift_base = fvecs_read("E:\\DATA\\vector_data\\sift1m\\sift_base.fvecs")
    sift_query = fvecs_read("E:\\DATA\\vector_data\\sift1m\\sift_query.fvecs")

    save_file = "E:\DATA\vector_data\sift100"
    label = np.random.randint(0, 1000000, size=[100])
    print(label)
