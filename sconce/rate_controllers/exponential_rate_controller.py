from sconce.rate_controllers.base import RateController

import numpy as np


class ExponentialRateController(RateController):
    """
    A Learning rate that rises exponentially from <min_learning_rate>
    to <max_learning_rate>, over <num_steps>.
    """
    def __init__(self, min_learning_rate, max_learning_rate, stop_factor=None,
            loss_key='training_loss'):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        self.stop_factor = stop_factor
        self.loss_key = loss_key
        self.min_loss = None

        self.learning_rates = None

    def start_session(self, num_steps):
        log_rates = np.linspace(np.log(self.min_learning_rate),
                                np.log(self.max_learning_rate),
                                num_steps)
        self.learning_rates = np.exp(log_rates)

    def new_learning_rate(self, step, data):
        if self.learning_rates is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > len(self.learning_rates):
            raise RuntimeError(f"Argument step={step}, should not "
                    f"exceed num_steps={len(self.learning_rates)}")

        if self.should_continue(data):
            new_learning_rate = self.learning_rates[step - 1]
            return new_learning_rate
        else:
            return None

    def should_continue(self, data):
        if self.loss_key not in data:
            return True

        loss = data[self.loss_key]
        try:
            loss = loss.item()
        except ValueError:
            loss = loss.data[0]

        if self.min_loss is None or loss < self.min_loss:
            self.min_loss = loss

        if (self.stop_factor is not None and
                loss > self.min_loss * self.stop_factor):
            return False

        return True
