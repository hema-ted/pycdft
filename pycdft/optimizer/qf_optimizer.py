import numpy as np
from pycdft.optimizer.base import Optimizer


class QFOptimizer(Optimizer):
    def __init__(self, x0, step=0.01, max_fit=5):
        super(QFOptimizer, self).__init__(x0)
        assert max_fit > 2
        self.step = step
        self.max_fit = max_fit

    def update(self, y, dydx):
        super(QFOptimizer, self).update(y, dydx)

        if len(self.xs) < self.max_fit:
            x_new = self.xs[-1] + self.step * dydx
        else:
            x_fit = self.xs[-self.max_fit:]
            y_fit = self.ys[-self.max_fit:]

            z = np.polyfit(x_fit, y_fit, 2)
            p = np.poly1d(z)
            a, b, _ = p.c
            x_new = -b / (2 * a)

        self.xs.append(x_new)
        return x_new
