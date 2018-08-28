import re
import numpy as np

def regex(dtype):
    """Returns the regular expression required by re package

    :param dtype: int, float or str
    :return: string of regular expression
    """

    if dtype is int:
        return r"-*\d+"
    elif dtype is float:
        return r"-*\d+\.\d*[DeEe]*[+-]*\d*"
    elif dtype is str:
        return r".*"
    else:
        raise ValueError("unsupported type")


def parse_many_values(n, dtype, content):
    """Parse n values of type dtype from content

    :param n: # of values wanted
    :param dtype: type of values wanted
    :param content: a string or a list of strings,
        it is assumed that n values exist in continues
        lines of content starting from the first line
    :return: a list of n values
    """

    if isinstance(content, str) or isinstance(content, np.string_):
        results = re.findall(regex(dtype), content)
        return [dtype(value) for value in results[0:n]]

    results = list()
    started = False
    for i in range(len(content)):
        found = re.findall(regex(dtype), content[i])
        if found:
            started = True
        else:
            if started:
                raise ValueError("cannot parse {} {} variables in content {}".format(
                    n, dtype, content
                ))
        results.extend(found)
        assert len(results) <= n
        if len(results) == n:
            return [dtype(result) for result in results]
