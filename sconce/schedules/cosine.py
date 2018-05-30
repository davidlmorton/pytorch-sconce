from sconce.schedules.base import Schedule

import math
import numpy as np


class Cosine(Schedule):
    """
    A Schedule where the hyperparameter follows a scaled and shifted cosine function
    from [0, pi].  It will begin at <initial_value> and end at <final_value>, after <num_steps>.
    """
    def __init__(self, initial_value, final_value, num_steps=2):
        self.initial_value = initial_value
        self.final_value = final_value
        self.set_num_steps(num_steps)

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
        self.progressions = np.linspace(0, 1, num_steps)

    def _get_value(self, step, current_state):
        progression = self.progressions[step - 1]

        new_value = (self.initial_value +
                (self.final_value - self.initial_value) *
                (1 + math.cos(math.pi * progression)) / 2)
        return new_value
