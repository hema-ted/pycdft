import numpy as np
from pycdft.optimizer.base import Optimizer


class QFOptimizer(Optimizer):
    def __init__(self, step=0.01, max_fit=5):
        assert max_fit > 2
        self.step = step
        self.max_fit = max_fit
        self.x_list = None
        self.y_list = None

    def setup(self):
        self.x_list = []
        self.y_list = []

    def update(self, dy_by_dx, x, y):
        self.x_list.append(x)
        self.y_list.append(y)

        if len(self.x_list) < self.max_fit:
            x_new = self.x_list[-1] + self.step * dy_by_dx
        else:
            x_fit = self.x_list[-self.max_fit:]
            y_fit = self.y_list[-self.max_fit:]

            z = np.polyfit(x_fit, y_fit, 2)
            p = np.poly1d(z)
            a, b, _ = p.c
            x_new = -b / (2 * a)

        return x_new
