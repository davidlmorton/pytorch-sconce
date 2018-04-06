from sconce.rate_controllers.base import RateController

import numpy as np


class TriangleRateController(RateController):
    """
    A Learning Rate that rises linearly from <min_learning_rate> to
    <max_learning_rate>, over <num_steps>/2 then drops linearly back to
    <min_learning_rate> over the remaining <num_steps>/2.
    """
    def __init__(self, min_learning_rate, max_learning_rate):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

        self.learning_rates = None

    def start_session(self, num_steps):
        # all fractions round up (instead of truncate)
        rise_steps = -(-num_steps // 2)

        rising_rates = np.linspace(self.min_learning_rate,
                                self.max_learning_rate,
                                rise_steps)
        falling_rates = np.linspace(self.max_learning_rate,
                                self.min_learning_rate,
                                (num_steps + 1) - rise_steps)
        self.learning_rates = np.concatenate((rising_rates,
                falling_rates[1:]))

    def new_learning_rate(self, step, data):
        if self.learning_rates is None:
            raise RuntimeError("You must call 'start_session' before calling "
                    "'new_learning_rate'")
        if step > len(self.learning_rates):
            raise RuntimeError(f"Argument step={step}, should not "
                    f"exceed num_steps={len(self.learning_rates)}")

        return self.learning_rates[step - 1]
