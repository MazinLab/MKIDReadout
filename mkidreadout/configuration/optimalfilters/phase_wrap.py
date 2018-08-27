import numpy as np


def fix_phase_wrap(data):
    """
    Fixes phase wrapping by assuming that jumps greater than pi radians are not real.
    :param data: mxn array where the rows are traces and the columns are different times
    :return: returns a new array of the same shape that has been modified
    """
    data_shape = data.shape
    new_data = np.atleast_2d(data.copy())
    differences = np.diff(new_data, axis=1)
    need_fix = np.abs(differences) > 1.2 * np.pi
    signs = -np.sign(differences)

    new_data[:, 1:] += 2 * np.pi * np.cumsum(need_fix * signs, axis=1)

    if len(data_shape) == 1:
        new_data = new_data.flatten()

    return new_data
