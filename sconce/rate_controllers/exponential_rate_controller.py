from sconce.rate_controllers.base import RateController

import numpy as np


class ExponentialRateController(RateController):
    """
    A Learning rate that rises exponentially from <min_learning_rate>
    to <max_learning_rate>, over <num_steps>.
    """
    def __init__(self, min_learning_rate, max_learning_rate):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

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
        if step >= len(self.learning_rates):
            raise RuntimeError(f"Argument step={step}, should not equal "
                    f"or exceed num_steps={len(self.learning_rates)}")

        new_learning_rate = self.learning_rates[step]
        return new_learning_rate
