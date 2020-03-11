from .sample import Sample
from .ft import FFTGrid
from .fragment import Fragment


def timer(start, end):
    """ Helper function for timing. """
    hours, rem = divmod(end-start,3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))