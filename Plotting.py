__author__ = 'Aaron'
from math import *
import matplotlib.pyplot as plt
import numpy as np


def plot(func, constraints):
    x = np.arange(constraints[0], constraints[1], 0.001)
    y = [func(x1) for x1 in x]
    fig = plt.figure()

    ax1 = fig.add_subplot('111')
    ax1.plot(x, y, linewidth=2)
    ax1.plot(x, np.zeros(len(x)), 'k')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    #ax1.set_title("$f(x) = (6x-2)^2 sin(12x-4)$")

    fig.savefig("../Plots/function.pdf")

if __name__ == '__main__':
    def f(x):
        return pow(6*x - 2, 2) * sin(12*x - 4)
    plot(f, [0, 1])