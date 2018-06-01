from pprint import pformat as pf
from sconce.schedules.base import Schedule

import numpy as np


class Linear(Schedule):
    """
    A Schedule where the value begins with value <initial_value>, then changes linearly to <final_value>.

    Arguments:
        initial_value (float): the initial value of the hyperparameter.
        final_value (float): the final value of the hyperparameter.
    """
    def __init__(self, initial_value, final_value):
        self.initial_value = initial_value
        self.final_value = final_value

    def __repr__(self):
        return f'{self.__class__.__name__}(initial_value={pf(self.initial_value)}, final_value={pf(self.final_value)})'

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
        self.values = np.linspace(self.initial_value, self.final_value, self.num_steps)

    def _get_value(self, step, current_state):
        return self.values[step - 1]
