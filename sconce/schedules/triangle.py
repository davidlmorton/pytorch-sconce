from sconce.schedules.base import Schedule

import numpy as np


class Triangle(Schedule):
    """
    A Schedule where the value begins with value <initial_value>, then changes linearly to <peak_value>,
    and then back to <initial_value> by the end of <num_steps>.  Peak value will occur after <peak_steps>.

    Arguments:
        initial_value (float): the initial value of the hyperparameter.
        peak_value (float): the value of the hyperparameter at it's peak (maximum or minimum).
        num_steps (int, optional): the total number of steps over which the hyperparameter's value will be controlled.
        peak_steps (int, optional): the number of steps before the hyperparameter's value will become <peak_value>.
            If ``None``, then peak_steps will be set to num_steps divided by two rounded up.
    """
    def __init__(self, initial_value, peak_value, num_steps=3, peak_steps=None):
        if peak_steps is None:
            peak_steps = -(-num_steps // 2)
        self.peak_steps = peak_steps
        self.initial_value = initial_value
        self.peak_value = peak_value
        self.set_num_steps(num_steps)

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
        if self.peak_steps is None:
            peak_steps = -(-num_steps // 2)
        else:
            peak_steps = self.peak_steps

        pre_peak_values = np.linspace(self.initial_value, self.peak_value, peak_steps)
        post_peak_values = np.linspace(self.peak_value, self.initial_value, (num_steps + 1) - peak_steps)
        self.values = np.concatenate((pre_peak_values, post_peak_values[1:]))

    def _get_value(self, step, current_state):
        return self.values[step - 1]
