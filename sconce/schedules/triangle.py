from pprint import pformat as pf
from sconce.schedules.base import Schedule

import math
import numpy as np


class Triangle(Schedule):
    """
    A Schedule where the value begins with value <initial_value>, then changes linearly to <peak_value>,
    and then back to <initial_value> by the end of <num_steps>.  Peak value will occur after <peak_steps>.

    Arguments:
        initial_value (float): the initial value of the hyperparameter.
        peak_value (float): the value of the hyperparameter at it's peak (maximum or minimum).
        peak_fraction (float, optional): (0.0, 1.0) used to determine the number of steps before the hyperparameter's
            value will become <peak_value>.
    """
    def __init__(self, initial_value, peak_value, peak_fraction=0.5):
        self.initial_value = initial_value
        self.peak_value = peak_value
        self.peak_fraction = peak_fraction

    def __repr__(self):
        return (f'{self.__class__.__name__}(initial_value={pf(self.initial_value)}, '
                f'peak_value={pf(self.peak_value)}, peak_fraction={pf(self.peak_fraction)})')

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
        self.peak_steps = math.ceil(num_steps * self.peak_fraction)

        pre_peak_values = np.linspace(self.initial_value, self.peak_value, self.peak_steps)
        post_peak_values = np.linspace(self.peak_value, self.initial_value, (num_steps + 1) - self.peak_steps)
        self.values = np.concatenate((pre_peak_values, post_peak_values[1:]))

    def _get_value(self, step, current_state):
        return self.values[step - 1]
