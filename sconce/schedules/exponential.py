from pprint import pformat as pf
from sconce.exceptions import StopTrainingError
from sconce.schedules.base import Schedule

import numpy as np


class Exponential(Schedule):
    """
    A Schedule where the value, adjusts exponentially from <initial_value>
    to <final_value>, over <num_steps>.

    Arguments:
        initial_value (float): the initial value of the hyperparameter.
        final_value (float): the final value of the hyperparameter.
        stop_factor (float, optional): a StopTrainingError will be raised if <loss_key> rises to
            <stop_factor> above it's observed minimum value so far.
        loss_key (string, optional): the name of the quantity (in ``current_state``) to observe for <stop_factor>.
    """
    def __init__(self, initial_value, final_value, stop_factor=None, loss_key='training_loss'):
        self.initial_value = initial_value
        self.final_value = final_value

        self.stop_factor = stop_factor
        self.loss_key = loss_key

    def __repr__(self):
        return (f'{self.__class__.__name__}(initial_value={pf(self.initial_value)}, '
                f'final_value={pf(self.final_value)}, stop_factor={pf(self.stop_factor)}, '
                f'loss_key={pf(self.loss_key)})')

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps
        self.min_loss = None

        log_rates = np.linspace(np.log(self.initial_value),
                                np.log(self.final_value),
                                num_steps)
        self.values = np.exp(log_rates)

    def _get_value(self, step, current_state):
        if self.should_continue(current_state):
            new_value = self.values[step - 1]
            return new_value
        else:
            raise StopTrainingError("Exponential Schedule stop condition met.")

    def should_continue(self, current_state):
        if self.loss_key not in current_state:
            return True

        loss = current_state[self.loss_key]
        try:
            loss = loss.item()
        except ValueError:
            loss = loss.current_state[0]

        if self.min_loss is None or loss < self.min_loss:
            self.min_loss = loss

        if (self.stop_factor is not None and
                loss > self.min_loss * self.stop_factor):
            return False

        return True
