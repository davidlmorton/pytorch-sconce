from pprint import pformat as pf
from sconce.schedules.base import Schedule

import math
import numpy as np


class Cosine(Schedule):
    """
    A Schedule where the hyperparameter follows a scaled and shifted cosine function
    from [0, pi].  It will begin at <initial_value> and end at <final_value>, after <num_steps>.

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
        self.progressions = np.linspace(0, 1, num_steps)

    def _get_value(self, step, current_state):
        progression = self.progressions[step - 1]

        new_value = (self.final_value +
                (self.initial_value - self.final_value) *
                (1 + math.cos(math.pi * progression)) / 2)
        return new_value
