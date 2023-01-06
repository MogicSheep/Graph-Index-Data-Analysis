import numpy as np
import math
from matplotlib import pyplot as plt



if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = y = np.arange(start=-4, stop=4, step=0.1)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    Z[Z<0]=0
    ax.plot_surface(X,Y,Z,alpha=0.9, cstride=1, rstride = 1, cmap='rainbow')
    plt.show()
