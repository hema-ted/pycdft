from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod


class Optimizer(object):

    """ An optimizer that maximize a function y = y(x) given x and x'. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def setup(self):
        """ Set up the optimizer with initial x. """
        pass

    @abstractmethod
    def update(self, dy_by_dx, x_new, y_new):
        """ Compute a new x value given y'(x) at current x value. """
        pass

