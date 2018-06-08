from pprint import pformat as pf
from sconce.schedules.base import Schedule

import numpy as np


class Step(Schedule):
    """
    A Schedule where the value starts at <initial_value> and changes <num_changes> times over the
    course of <num_steps> to a final value of <final_value>.

    Arguments:
        initial_value (float): the initial value of the hyperparameter.
        final_value (float): the final value of the hyperparameter.
        num_changes (int, optional): [1, <num_steps>-1] the number of times the hyperparameter's value will change.

    Note:
        The parameter <num_steps> is set during training based on the size of the batch_size and number of samples in
        the training_feed, and the batch_multiplier value.
    """
    def __init__(self, initial_value, final_value, num_changes=1):
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_changes = num_changes

    def __repr__(self):
        return (f'{self.__class__.__name__}(initial_value={pf(self.initial_value)}, '
                f'final_value={pf(self.final_value)}, num_changes={pf(self.num_changes)})')

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

        num_regions = self.num_changes + 1
        self.region_values = np.linspace(self.initial_value,
                self.final_value, num_regions)
        self.indices = [int(i) for i in np.linspace(0, num_regions - 1e-12, num_steps)]

    def _get_value(self, step, current_state):
        return self.region_values[self.indices[step - 1]]
