__author__ = 'Aaron'
from random import *
from math import *


class GD:
    """
    Use a gradient descent algorithm in order to optimize a single or
    multidimensional function.
    """
    def __init__(self, function, first_deriv, constraints):
        self.function = function
        self.first_deriv = first_deriv
        self.constraints = constraints
        self.c_range = constraints[1] - constraints[0]
        self.step_size = 1.0                                 # Default use 1/1000 for step size

    # noinspection PyMethodMayBeStatic
    def optimize(self, _x0):
        if _x0 is None:
            x_hat = (random() * self.c_range) + self.constraints[0]
        else:
            x_hat = _x0
        while abs(self.first_deriv(x_hat)) >= 0.001 and self.within_bounds(x_hat):
            x_hat -= self.step_size * self.first_deriv(x_hat)
        return x_hat

    def within_bounds(self, x_val):
        return self.constraints[0] <= x_val <= self.constraints[1]


if __name__ == '__main__':
    # First test case
    def f(x):
        return pow(6*x-2, 2) * sin(12*x-4)

    def f_prime(x):
        return 12*pow(6*x-2, 2)*cos(4-12*x) - 12*(6*x-2)*sin(4-12*x)

    Problem1 = GD(f, f_prime, [0, 1])
    start = [0.05, 0.5, 0.6, 0.9]

    for x0 in start:
        print(x0)
        x = Problem1.optimize(x0)
        print("For the lab problem with x0 = " + str(x0) + ":\n\tmaximum (x, y) was: (" +
              format(x, ".5f") + ", " + format(f(x), ".5f") + ")")
