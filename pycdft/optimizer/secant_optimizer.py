import numpy as np
from pycdft.optimizer.base import Optimizer


class SecantOptimizer(Optimizer):
    def __init__(self, x0, dx0):
        super(SecantOptimizer, self).__init__(x0)
        self.dx0 = dx0

    def update(self, y, dydx):
        super(SecantOptimizer, self).update(y, dydx)

        if len(self.xs) >= 2:
            x_n2, x_n1 = self.xs[-2:]
            dydx_n2, dydx_n1 = self.dydxs[-2:]
            x_new = (x_n2 * dydx_n1 - x_n1 * dydx_n2) / (dydx_n1 - dydx_n2)
        else:
            x_new = self.xs[-1] + self.dx0 * np.sign(y)

        self.xs.append(x_new)
        return x_new
