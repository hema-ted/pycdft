from pycdft.optimizer.base import Optimizer


class SDOptimizer(Optimizer):
    def __init__(self, x0, dx=0.01):
        super(SDOptimizer, self).__init__(x0)
        self.dx = dx

    def update(self, y, dydx):
        super(SDOptimizer, self).update(y, dydx)
        x_new = self.xs[-1] + self.dx * dydx
        self.xs.append(x_new)
        return x_new
