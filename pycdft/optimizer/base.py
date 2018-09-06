from abc import ABCMeta, abstractmethod


class Optimizer(object):

    """ An optimizer that maximize a function y = y(x) given x and x'. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, dx0):
        self.xs = [dx0]
        self.ys = []
        self.dydxs = []

    def reset(self):
        """ Reset the optimizer. """
        self.xs = self.xs[-1:]
        self.ys = []
        self.dydxs = []

    @abstractmethod
    def update(self, y, dydx):
        """ Compute a new x value given y and y'(x) at current x value. """
        self.ys.append(y)
        self.dydxs.append(dydx)

